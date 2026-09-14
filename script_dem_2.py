import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt  # 超高速影像距離變換
from scipy.ndimage import gaussian_filter         # 【新導入】高斯濾波器
import rasterio
from rasterio.transform import from_origin
import matplotlib.path as mpath                   
import json
import time                                       

# ==========================================
# 1. 參數設定與資料載入
# ==========================================
INPUT_FILE = "demo_xyz.csv"
ROI_FILE = "roi.geojson"                         
OUTPUT_TIFF = INPUT_FILE[:-4] + ".tif"
OUTPUT_OUTLIERS_CSV = INPUT_FILE[:-4] + "_Outliers.csv"  

RESOLUTION = 10.0          
BOUNDARY_Z = -1.0          

SEA_BUFFER_RADIUS = 450.0   # 限制 800m 單音束測線外推的安全半徑
LAND_SEARCH_RADIUS = 40.0   # 陸域 IDW 搜尋半徑
IDW_POWER = 2.0            
OUTLIER_THRESHOLD = 0.5    

# 【高斯平滑核心參數】
# SMOOTH_THRESHOLD_Z: 低於此高程(水深)才進行平滑，避免破壞潮間帶與陸域細節
# GAUSSIAN_SIGMA: 標準差。數值越大越平滑。1.5~2.5 像素最適合消除 800m 測線間的四邊形
SMOOTH_THRESHOLD_Z = -5.0  
GAUSSIAN_SIGMA = 3       

print("正在讀取 roi.geojson 範圍...")
with open(ROI_FILE, 'r', encoding='utf-8') as f:
    geojson = json.load(f)

# 支援處理多種不同的 GeoJSON 結構
feature = geojson['features'][0] if 'features' in geojson else geojson
geom = feature['geometry'] if 'geometry' in feature else geojson['geometry']

if geom['type'] == 'Polygon':
    roi_vertices = np.array(geom['coordinates'][0])
elif geom['type'] == 'MultiPolygon':
    roi_vertices = np.array(geom['coordinates'][0][0])
else:
    raise ValueError("僅支援 Polygon 或 MultiPolygon 格式的 ROI")

roi_path = mpath.Path(roi_vertices)

print("正在載入原始點雲資料...")
df_raw = pd.read_csv(INPUT_FILE, sep=r'\s+', names=['X', 'Y', 'Z'], dtype=np.float64)
print(f"原始點雲總數: {len(df_raw)} 點")

print("正在利用 ROI 範圍裁剪點雲資料 (PIP 篩選)...")
pts_xy = df_raw[['X', 'Y']].values
inside_roi = roi_path.contains_points(pts_xy)
df = df_raw[inside_roi].copy()
print(f"ROI 裁剪後剩餘點數: {len(df)} 點")

if len(df) == 0:
    raise ValueError("錯誤：ROI 範圍內沒有任何實測點雲資料，請檢查坐標系統是否對齊。")

# 分層隨機抽樣 5% 作為「獨立檢核點」
np.random.seed(42)
df_land = df[df['Z'] >= BOUNDARY_Z]
df_sea = df[df['Z'] < BOUNDARY_Z]

mask_check_land = np.random.rand(len(df_land)) < 0.05
mask_check_sea = np.random.rand(len(df_sea)) < 0.05

df_check = pd.concat([df_land[mask_check_land], df_sea[mask_check_sea]])
df_model = pd.concat([df_land[~mask_check_land], df_sea[~mask_check_sea]])

X_pts = df_model['X'].values
Y_pts = df_model['Y'].values
Z_pts = df_model['Z'].values

# 定義精簡後的網格範圍
x_min, x_max = X_pts.min(), X_pts.max()
y_min, y_max = Y_pts.min(), Y_pts.max()
grid_x = np.arange(x_min, x_max + RESOLUTION, RESOLUTION)
grid_y = np.arange(y_max, y_min - RESOLUTION, -RESOLUTION) 
grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

out_dem = np.full(grid_X.shape, np.nan, dtype=np.float32)

# 分離海陸建模群組
land_mask_pts = Z_pts >= BOUNDARY_Z
sea_mask_pts = Z_pts < BOUNDARY_Z

# ==========================================
# 2. 陸域建模：KDTree + IDW
# ==========================================
print("\n[開始] 執行陸域高密度點雲內插 (KDTree + IDW)...")
start_land_time = time.perf_counter()

land_coords = np.column_stack((X_pts[land_mask_pts], Y_pts[land_mask_pts]))
land_z = Z_pts[land_mask_pts]

if len(land_coords) > 0:
    land_tree = KDTree(land_coords)
    grid_flat_coords = np.column_stack((grid_X.ravel(), grid_Y.ravel()))
    
    distances, indices = land_tree.query(grid_flat_coords, k=4, distance_upper_bound=LAND_SEARCH_RADIUS)
    
    idw_vals = np.full(len(grid_flat_coords), np.nan)
    for i in range(len(grid_flat_coords)):
        valid = (distances[i] > 0) & (distances[i] < LAND_SEARCH_RADIUS) & (indices[i] < len(land_z))
        exact = distances[i] == 0
        if np.any(exact):
            idw_vals[i] = land_z[indices[i][exact]]
        elif np.any(valid):
            w = 1.0 / (distances[i][valid] ** IDW_POWER)
            idw_vals[i] = np.sum(w * land_z[indices[i][valid]]) / np.sum(w)
            
    land_fill_mask = (idw_vals >= BOUNDARY_Z)
    out_dem.flat[land_fill_mask] = idw_vals[land_fill_mask]

end_land_time = time.perf_counter()
land_elapsed = end_land_time - start_land_time

# ==========================================
# 3. 海域建模：TIN + 超高速距離變換 (防外插)
# ==========================================
print("[開始] 執行海域單音束內插 (TIN + 影像級距離變換)...")
start_sea_time = time.perf_counter()

sea_coords = np.column_stack((X_pts[sea_mask_pts], Y_pts[sea_mask_pts]))
sea_z = Z_pts[sea_mask_pts]

if len(sea_coords) > 3:
    interp_tin = LinearNDInterpolator(sea_coords, sea_z)
    dem_sea_tin = interp_tin(grid_X, grid_Y)
    
    line_mask = np.ones(grid_X.shape, dtype=bool)
    
    # 修正廣播錯誤，明確減去陣列的首個元素 (Index 0)
    pixel_cols = np.round((sea_coords[:, 0] - grid_x[0]) / RESOLUTION).astype(int)
    pixel_rows = np.round((grid_y[0] - sea_coords[:, 1]) / RESOLUTION).astype(int)
    
    valid_pts = (pixel_cols >= 0) & (pixel_cols < grid_X.shape[1]) & \
                (pixel_rows >= 0) & (pixel_rows < grid_X.shape[0])
    line_mask[pixel_rows[valid_pts], pixel_cols[valid_pts]] = 0
    
    sea_distances_2d = distance_transform_edt(line_mask) * RESOLUTION
    
    sea_valid_mask = (dem_sea_tin < BOUNDARY_Z) & (np.isnan(out_dem) | (out_dem < BOUNDARY_Z)) & (sea_distances_2d <= SEA_BUFFER_RADIUS)
    out_dem[sea_valid_mask] = dem_sea_tin[sea_valid_mask]

end_sea_time = time.perf_counter()
sea_elapsed = end_sea_time - start_sea_time

# ==========================================
# 3.5 【全新優化區功能】海域深水區動態高斯平滑
# ==========================================
print(f"[優化] 正在針對深水區 (Z < {SMOOTH_THRESHOLD_Z}m) 進行高斯平滑消除航線四邊形雜訊...")
start_smooth_time = time.perf_counter()

# 定義深水有效網格遮罩 (必須有數值且低於設定水深)
deep_sea_mask = (out_dem < SMOOTH_THRESHOLD_Z) & (~np.isnan(out_dem))

if np.any(deep_sea_mask):
    # 由於高斯濾波無法直接處理 NaN，先將 NaN 暫時填補為鄰近平均或 0
    # 並配合海域的有效範圍進行遮罩限縮
    dem_filled = np.copy(out_dem)
    nan_mask = np.isnan(dem_filled)
    dem_filled[nan_mask] = 0.0 # 暫時填 0，但後續會被遮罩保護
    
    # 執行二維高斯濾波，消除非自然折角
    smoothed_dem = gaussian_filter(dem_filled, sigma=GAUSSIAN_SIGMA)
    
    # 【關鍵嚴密揉合】：僅將平滑後的數值填回深海區，陸域、沿岸、NoData 空白區完全不受干擾
    out_dem[deep_sea_mask] = smoothed_dem[deep_sea_mask]

end_smooth_time = time.perf_counter()
smooth_elapsed = end_smooth_time - start_smooth_time

# ==========================================
# 4. 匯出高精度 GeoTIFF
# ==========================================
print("正在寫入 GeoTIFF 檔案...")
# 修正原點為 Scalar 值 (Index 0)
transform = from_origin(grid_x[0], grid_y[0], RESOLUTION, RESOLUTION)

with rasterio.open(
    OUTPUT_TIFF, 'w', driver='GTiff',
    height=out_dem.shape[0], width=out_dem.shape[1],
    count=1, dtype=out_dem.dtype, crs=rasterio.crs.CRS.from_epsg(3826),
    transform=transform, nodata=np.nan
) as dst:
    dst.write(out_dem, 1)

# ==========================================
# 5. 結果驗證與異常點導出模組
# ==========================================
print("\n" + "="*50)
print("          高精度地形水深結果驗證報告          ")
print("="*50)
print(f"【核心建模耗時統計】")
print(f"  - 陸域群組計算耗時 (IDW + KDTree) : {land_elapsed:.4f} 秒")
print(f"  - 海域群組計算耗時 (TIN + 距離變換): {sea_elapsed:.4f} 秒")
print(f"  - 後處理高斯平滑耗時             : {smooth_elapsed:.4f} 秒")
print("-" * 50)

check_x = df_check['X'].values
check_y = df_check['Y'].values
check_z = df_check['Z'].values

land_errors, sea_errors, total_errors = [], [], []
outlier_list = []  

with rasterio.open(OUTPUT_TIFF) as src:
    dem_raster = src.read(1)
    
    for x, y, z in zip(check_x, check_y, check_z):
        row, col = src.index(x, y)
        if 0 <= row < src.height and 0 <= col < src.width:
            pred_z = dem_raster[row, col]
            if not np.isnan(pred_z):
                error = float(pred_z - z)
                total_errors.append(error)
                if z >= BOUNDARY_Z:
                    land_errors.append(error)
                else:
                    sea_errors.append(error)
                
                if abs(error) > OUTLIER_THRESHOLD:
                    outlier_list.append({
                        'X': x, 'Y': y, 'True_Z': z, 'Pred_Z': float(pred_z),
                        'Absolute_Error': abs(error), 'Error': error,
                        'Region': 'Land' if z >= BOUNDARY_Z else 'Sea'
                    })

total_errors = np.array(total_errors)
land_errors = np.array(land_errors)
sea_errors = np.array(sea_errors)

def calculate_metrics(err_array):
    if len(err_array) == 0: return (0.0, 0.0, 0.0)
    me = np.mean(err_array)
    mae = np.mean(np.abs(err_array))
    rmse = np.sqrt(np.mean(err_array ** 2))
    return me, mae, rmse

me_t, mae_t, rmse_t = calculate_metrics(total_errors)
me_l, mae_l, rmse_l = calculate_metrics(land_errors)
me_s, mae_s, rmse_s = calculate_metrics(sea_errors)

print(f"ROI 區域內參與驗證之有效檢核點共：{len(total_errors)} 點")
print(f"【整體海陸綜合精度】")
print(f"  - 平均誤差 (Mean Error)      : {me_t:+.3f} 公尺")
print(f"  - 平均絕對誤差 (MAE)         : {mae_t:.3f} 公尺")
print(f"  - 均方根誤差 (RMSE)          : {rmse_t:.3f} 公尺")
print("-" * 50)
print(f"【陸域區精細度 (RMSE)】        : {rmse_l:.3f} 公尺 (點數: {len(land_errors)})")
print(f"【海域平滑區精度 (RMSE)】      : {rmse_s:.3f} 公尺 (點數: {len(sea_errors)})")
print("-" * 50)

print(f"【異常點 (Outliers) 分析】(判定標準: 絕對誤差 > {OUTLIER_THRESHOLD}m)")
print(f"  - 偵測到異常檢核點數量       : {len(outlier_list)} 點")

if len(outlier_list) > 0:
    df_outliers = pd.DataFrame(outlier_list)
    df_outliers = df_outliers.sort_values(by='Absolute_Error', ascending=False)
    df_outliers.to_csv(OUTPUT_OUTLIERS_CSV, index=False, encoding='utf-8-sig')
    print(f"  -> 異常點明細已成功導出至：{OUTPUT_OUTLIERS_CSV}")
else:
    print("  -> 未偵測到任何超出閾值的異常點。")
print("="*50)
