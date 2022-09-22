# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 02:57:12 2021

@author: xurui
"""
import shapefile
import geopandas as gpd
import rasterio
from osgeo import gdal
from shapely import geometry
from rasterio.mask import mask
#%%

def create_sf(new_outline, lats, longs, f_date):
    geo_sp = []
    for n in range(new_outline.shape[0]):
        if n%20 ==0:
            i,j =new_outline[:,0][n],new_outline[:,1][n]
            lat, long = lats[j,i], longs[j,i]
            #plt.scatter(long, lat)
            geo_sp.append((long,lat))
    cq = gpd.GeoSeries([geometry.Polygon(geo_sp)],
    index=['Salton Sea'], crs='EPSG:32611', # coordinate system WGS 1984
    )
    
    cq.to_file(f'D:\\MSDA\\data298\\Shapefiles\\SS{f_date}.shp', driver='ESRI Shapefile',
    encoding='utf-8')
def get_dist_ss(date_sf):
    f = open(f'D:\\MSDA\\data298\\Shapefiles\\SS{date_sf}.geojson')
    data = json.load(f)
    #print(f'D:\\MSDA\\data298\\Shapefiles\\SS{date_sf}.geojson loading')
    lats = np.load(r'd:\msda\data298\new\lats.npy')
    longs = np.load(r'd:\msda\data298\new\longs.npy')
    coors = data['features'][0]['geometry']['coordinates'][0]
    rowcol = []
    for i in coors:
        long,lat = i
        row, col = get_rowcol(lats, longs, lat, long)
        rowcol.append([row,col])
    empty_m = np.zeros((109,197))
    dist_to_ss = empty_m.copy()
    for r in range(empty_m.shape[0]):
        for c in range(empty_m.shape[1]):
            dist_min = np.min([((r-i[0])**2+(c-i[1])**2)**.5 for i in rowcol])
            dist_to_ss[r,c] = dist_min
    print(1)
    rowcol=np.array(rowcol)
    #plt.scatter(rowcol[:,1],rowcol[:,0])
    #plt.imshow(dist_to_ss, cmap='gray_r')
    #plt.show()
    return dist_to_ss
#%%

files=glob.glob('D:\MSDA\data298\Salton sea\*.tif')
for n in range(0, len(files)):
    r, lats, longs = read_tif(files[n])
    f_date = files[n].split('039037_')[1][:8]
    
    img = cv2.imread(files[n], cv2.IMREAD_UNCHANGED)
    #plt.imshow(img)
    img = cv2.convertScaleAbs(img)# remember to convert the data type from float64 to int8 or float32. or youll get error
    img = cv2.GaussianBlur(img, (5, 5), 10)
    #plt.imshow(img)
    masked=np.where(img!=int(img[3000,3000]), 255, img)#[2500:4500,2000:4500]
    #plt.imshow(masked)
    ret, binary = cv2.threshold(masked,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), 'contours')
    #cv2.drawContours(masked,contours,-1,(0,255,0),30)
    h, w = masked.shape
    threshold = h/30 * w/30 
    suma=0
    areas=[]
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) 
        #print(area, i, end=('  \  '))
        suma+=area
        areas.append(area)
        if area < threshold:
            cv2.drawContours(masked,[contours[i]],-1, (255), thickness=-1) 
            continue
    contour_idx = np.argsort(areas)[-2] # get the second largest contour. for the largest contour is the background
    if areas[contour_idx]<890000: # because of cloud, some parts of waterbody is not detected as water. skip this tif.
        print(areas[contour_idx])
        plt.imshow(masked)
        plt.show()
    else:
        outline = np.array(contours[contour_idx])
        outline.shape
        new_outline=outline.transpose(1,0,2)[0] # the shape is (points, 1, 2) 2 for row&col. transpose to (points, 2)
        
        plt.plot(new_outline[:,0], new_outline[:,1])
        plt.imshow(masked)
        plt.show()
        create_sf(new_outline, lats, longs, f_date)
        print('Succeed creating shapefile', f_date)


f_date='20191226'

ss = gpd.read_file(f'D:\\MSDA\\data298\\Shapefiles\\SS{f_date}.shp')
ss.plot()



import geopandas
sffiles=glob.glob('D:\\MSDA\\data298\\Shapefiles\\*.shp')
for i in sffiles[4:]:
    f_n = i.split('SS')[1][:8]
    shp_file = geopandas.read_file(i)
    shp_file.to_file(f'D:\\MSDA\\data298\\Shapefiles\\SS{f_n}.geojson', driver='GeoJSON')
    
    
    
    sfs = glob.glob('D:\\MSDA\\data298\\Shapefiles\\*.geojson')    
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    date_ = date_obj.strftime('%Y%m%d')
    date_sf = [str(i.split('SS')[1][:8]) for i in sfs]
    date_sf_d = [abs(datetime.strptime(d, '%Y%m%d')-date_obj) for d in date_sf]

    date_of_sf = date_sf[np.argmin(date_sf_d)]




input_f = r'D:\MSDA\data298\MODIS\all_aod\cropped\MCD19A2.A2018336.h08v05.006.2018338033720_055_cropped.tif'
r, lats, longs = read_tif(input_f)
f = open(f'D:\\MSDA\\data298\\Shapefiles\\SS{date}.geojson')
data = json.load(f)
coors = data['features'][0]['geometry']['coordinates'][0]
rowcol = []
for i in coors:
    long,lat = i
    row, col = get_rowcol(lats, longs, lat, long)
    rowcol.append([row,col])
empty_m = np.zeros((109,197))
dist_to_ss = empty_m.copy()
for r in range(empty_m.shape[0]):
    for c in range(empty_m.shape[1]):
        dist_min = np.min([((r-i[0])**2+(c-i[1])**2)**.5 for i in rowcol])
        print(r,c, dist_min)
        dist_to_ss[r,c] = dist_min
rowcol=np.array(rowcol)
plt.scatter( rowcol[:,1],rowcol[:,0])
plt.imshow(empty_m, cmap='gray_r')
plt.show()
return dist_to_ss
#%%


df25 = pd.read_csv('d:\msda\data298\epa\daily\pm2.5SS.csv')
df10 = pd.read_csv('d:\msda\data298\epa\daily\pm10SS.csv')
for r in range(empty_m.shape[0]):
    for c in range(empty_m.shape[1]):
        dist_min = abs(r  - 42)+ abs(c - 103)
        print(r,c, dist_min)
        empty_m[r,c] = dist_min
        
        df = df10.copy()
df['tmp'] = np.nan
df['dist'] = 0
sfs = glob.glob('D:\\MSDA\\data298\\Shapefiles\\*.geojson')   
df_d = df[['Date Local']].drop_duplicates().reset_index(drop=True)
for i in df_d['Date Local'].values:
    date_obj = datetime.strptime(i, '%Y-%m-%d')
    date_ = date_obj.strftime('%Y%m%d')
    date_sf = [str(i.split('SS')[1][:8]) for i in sfs]
    date_sf_d = [abs(datetime.strptime(d, '%Y%m%d')-date_obj) for d in date_sf]
    date_of_sf = date_sf[np.argmin(date_sf_d)]
    df.loc[df['Date Local']==i,'tmp'] = date_of_sf

        
sfs = glob.glob('D:\\MSDA\\data298\\Shapefiles\\*.geojson')   
df_d = df[['tmp']].drop_duplicates().reset_index(drop=True)
lats = np.load(r'd:\msda\data298\new\lats.npy')
longs = np.load(r'd:\msda\data298\new\longs.npy')
for sfd in df_d['tmp'].values:
    print(sfd)
    dist_m = get_dist_ss(sfd)
    dftmp = df.loc[df['tmp']==sfd][['Latitude','Longitude']].drop_duplicates().reset_index(drop=True)
    for i in range(len(dftmp)):
        lat = dftmp.loc[i, 'Latitude']
        long = dftmp.loc[i, 'Longitude']
        row, col = get_rowcol(lats, longs, lat, long)
        print(len(df.loc[(df['tmp']==sfd)&(df['Latitude']==lat)&(df['Longitude']==long)]),row,col,dist_m[row,col])
        df.loc[(df['tmp']==sfd)&(df['Latitude']==lat)&(df['Longitude']==long), 'dist'] = round(dist_m[row,col],2)