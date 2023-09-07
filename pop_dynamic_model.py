import osmnx as ox
import math
import pyproj
import os
import numpy as np
import shapely as shp
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm

by_type = {'str':'string', 'list':'object', 'int':'Int64',
           'nan': 'string', 'float':'Float64', 'object':'object', 'bool':'boolean'}

by_name = {'building': 'category', 'source': 'string',
  'highway': 'category',
  'addr:housenumber': 'string',
  'addr:street': 'string',
  'addr:city': 'string',
  'name': 'string',
  'addr:postcode': 'string',
  'natural': 'category',
  'surface': 'category',
  'addr:country': 'string',
  'landuse': 'category',
  'power': 'category',
  'building:levels': 'Int64',
  'waterway': 'category',
  'amenity': 'category',
  'service': 'category',
  'barrier': 'category',
  'addr:state': 'string',
  'access': 'category',
  'public_transport':'category'}

def process_bar(mass, count=0, text=''):
    # чистка строки состояния
    print(' ' * len(text), end='\r')
    
    # обновление строки состояния
    count += 1
    percent = count / mass * 100
    text = f'Обработано {count} из {mass}: {round(percent, 2)}%'
    print(text, end='\r', flush=True)
    
    return count

# функция возвращает utm по географическим координатам (широта, долгота)
def convert_wgs_to_utm(lat: float, lon: float):
    """
    Возвращает наиболее подходящую проекцию UTM по принимаемым координатам широты и долготы.

    Принимает
    ----------
    lat: float      координата широты.
    lon: float      координата долготы.

    Возвращает
    -------
    CRS             объект класса CRS модуля pyproj.
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    prop_crs = pyproj.CRS.from_epsg(epsg_code)
    return prop_crs

def recreate(gf):
    gdf = gf.gdf.copy()
    query = gf.query
    gf = Hex(gdf)
    gf.query = query

class GeoFrame:
    name = 'gdf'
    dir_path = '.'

    def __init__(self, gdf=gpd.GeoDataFrame()):
        self.gdf = gdf
        self._set_dtypes_by_types()
        self._set_dtypes_by_names()

    def _set_dtypes_by_types(self):
        cols_types = {}
        for col_name in self.gdf.columns:
            if col_name == 'geometry':
                continue
            col = self.gdf[col_name].dropna()
            col = col.apply(lambda x: type(x).__name__)
            col_types = col.unique().tolist()
            if len(col_types) == 0:
                col_type = 'nan'
            elif len(col_types) == 1:
                col_type = col_types[0]
            else:
                col_type = 'object'
            cols_types[col_name] = by_type[col_type]    
        self.gdf = self.gdf.astype(cols_types)
        
    def _set_dtypes_by_names(self):
        cols_types = {}
        for col_name in self.gdf.columns:
            if col_name in by_name:
                cols_types[col_name] = by_name[col_name]        
        for col_name in cols_types:
            if cols_types[col_name] in ('Int64', 'Float64'):
                col = pd.to_numeric(self.gdf[col_name], errors='coerce')
                if cols_types[col_name] == 'Int64':
                    col = col.round()
                self.gdf[col_name] = col
            self.gdf[col_name] = self.gdf[col_name].astype(cols_types[col_name])
    
    @classmethod
    def from_file(cls, file_path):
        # читаем gdf из файла и создаём экземпляр класса
        gdf = gpd.read_file(file_path)
        gf = cls(gdf)

        return gf
    
    # метод позволяет выгружать геодатафрейм по названию
    @classmethod
    def by_query(cls, query):
        # задаём геодатафрейм
        gdf = ox.geocode_to_gdf(query)
        
        # создаём экземпляр класса
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query

        return gf
    
    @property
    def lat(self, column_name='lat'):
        if column_name in self.gdf.columns:
            lat = self.gdf[column_name].mean()
        elif hasattr(self, '_query'):
            lat = ox.geocode(self._query)[0]
        else:
            lat = self.gdf.geometry.centroid.y.mean()

        return lat

    @property
    def lon(self, column_name='lon'):
        if column_name in self.gdf.columns:
            lon = self.gdf[column_name].mean()
        elif hasattr(self, '_query'):
            lon = ox.geocode(self._query)[1]
        else:
            lon = self.gdf.geometry.centroid.x.mean()

        return lon

    @property
    def crs(self):
        return self.gdf.crs

    @property
    def utm_crs(self):
        if self.crs.is_geographic:
            return convert_wgs_to_utm(self.lat, self.lon)
        else:
            return self.crs
    
    @property
    def area(self):
        if self.crs.is_geographic:
            gdf = self.gdf.to_crs(self.utm_crs)
        else:
            gdf = self.gdf
        return gdf.area
    
    @property
    def length(self):
        if self.crs.is_geographic:
            gdf = self.gdf.to_crs(self.utm_crs)
        else:
            gdf = self.gdf
        return gdf.length
            
    @property
    def query(self):
        return self._query
    
    @query.setter
    def query(self, query):
        if query is None:
            return
        self._query = query
        self.query_info = ox.geocode_to_gdf(self._query)
        self.name = self.query_info.display_name[0].split(',')[0]
        key = self.query_info['class'][0]
        value = self.query_info['type'][0]
        self.dir_path = f'query/{key}/{value}/{__class__.dir_path}'

    
    def simplify(self, tolerance=1, inplace=False):
        gdf = self.gdf.copy()
        gdf = gdf.to_crs(self.utm_crs)
        
        gdf = gdf.geometry.simplify(tolerance)
        gdf = gdf.to_crs(self.crs)
        
        if inplace:
            self.gdf.geometry = gdf
        
        return gdf
    
    
    def merge_line_geometry(self, inplace=False):
        gdf = self.gdf.copy()
        
        gdf = gdf[gdf.geometry.apply(lambda x: type(x).__name__)=='LineString']
        
        count = 0
        trash = {1}
        while len(trash)>0:
            
            count = 0
            trash = set()
            
            for i in gdf.index:
                 
                # некоторые индексы в процессе выполнения пропадают, проверка
                if i in trash:
                    count = process_bar(len(gdf), count)
                    continue
                
                # исследуемая линия
                line = gdf.loc[i,'geometry']
                
                # линии, касающиеся этих точек
                touch = shp.touches(line, gdf.geometry)
                touch = touch[touch==True]
                
                # если касаний нет, пропуск
                if len(touch)==0:
                    count = process_bar(len(gdf), count)
                    continue
                
                # непосредственно список касающихся линий (+ исходная)
                touch_lines = gdf.loc[touch.index].geometry.tolist()
                touch_lines.append(gdf.loc[i,'geometry'])
                
                # связываем
                LINE = shp.geometry.MultiLineString(touch_lines)
                LINE = shp.line_merge(LINE)
                
                # если не связалось, пропуск
                if type(LINE).__name__=='MultiLineString':
                    count = process_bar(len(gdf), count)
                    continue
                
                # присваиваем исходной линии новую геометрию
                gdf.loc[i,'geometry'] = LINE
                
                # добавляю присоединённые линии в мусорку
                trash.update(list(touch.index))
                
                # счётчик
                count = process_bar(len(gdf), count)
            
            # во сколько раз уменьшился датасекс 
            sqz = len(gdf) / (len(gdf) - len(trash))
            
            text = f'\nСжато {len(trash)}/{len(gdf)}: в {round(sqz, 2)} раз'
            print(text)
            print('_' * len(text))
            
            # убираем мусор
            gdf.drop(trash, inplace=True)

        sqz = len(self.gdf) / len(gdf)
        text = f'Итого сжато в {round(sqz, 2)} раз'
        print(text)        
        
        if inplace:
            self.gdf = gdf
        
        return gdf
    
        
    # экспорт геодатафрейма в нужном формате
    def export(self, formate='geojson', *, name=None, dir_path=None):
        """
        Экспортирует объект класса geopandas.GeoDataFrame в файл указанного формата.

        Принимает
        ----------
        frame           объект класса geopandas.GeoDataFrame.
        name: str       (опционально) имя файла; по умолчанию gdf.
        dir_path: str   (опционально) путь, если файл нужно разместить в определённой папке; создаёт папку, если её нет.
        formate: str    (опционально) формат файла; по умолчанию csv.
        """
        name = self.name if name is None else name
        dir_path = self.dir_path if dir_path is None else dir_path

        os.makedirs(dir_path, exist_ok=True)
        file_path = f'{dir_path}/{name}.{formate}'
        gdf = self.gdf.copy()

        # непосредственно экспорт
        if formate == 'csv':
            gdf.to_csv(file_path, encoding='utf_8_sig')
        elif formate == 'geojson':
            cols_to_drop = gdf.select_dtypes(include=['object']).columns
            gdf.drop(columns=cols_to_drop, inplace=True)
            cols_to_change = gdf.select_dtypes(include=['category']).columns
            gdf[cols_to_change] = gdf.select_dtypes(include=['category']).astype('string')
            cols_to_change = gdf.select_dtypes(include=['Float64']).columns
            gdf[cols_to_change] = gdf.select_dtypes(include=['Float64']).astype(float)
            gdf.reset_index(inplace=True)
            gdf.to_file(file_path, 'GeoJSON')


class Place(GeoFrame):
    dir_path = 'place'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def by_query(cls, query):
        gdf = ox.geocode_to_gdf(query)
        
        place_id = (gdf['osm_type'][0], gdf['osm_id'][0])
        place_class = gdf['class'][0]
        place_type = gdf['type'][0]
        
        gdf = ox.features_from_place(query, {place_class: place_type})
        gdf = gdf.loc[[place_id]]
        gdf.reset_index(inplace=True)
        
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query
        
        return gf


class Buildings(GeoFrame):
    dir_path = 'buildings'

    types = {
        'accommodation': [
            'apartments', 'barracks', 'bungalow', 'cabin',
            'detached', 'dormitory', 'farm', 'ger', 'hotel',
            'house', 'houseboat', 'residential', 'semidetached_house',
            'static_caravan', 'stilt_house', 'terrace', 'tree_house'
        ]

    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # функция получения геометрии требуемых типов зданий (по умолчанию выводятся все здания)
    @classmethod
    def by_query(cls, query, types=True):
        gdf = ox.features_from_place(query, {'building': types})
        gdf = gdf.loc[gdf.geometry.geom_type == 'Polygon']
        
        gdf.reset_index(inplace=True)
        
        gdf.sort_index(inplace=True)
        
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query
        
        return gf

    def complete_levels_by_type(self, inplace=False):
        gdf = self.gdf[['building', 'building:levels']].copy()
        
        # в осм не дана этажность всех зданий, поэтому как костыль берётся
        # средняя этажность по каждому типу зданий, которая ставится вместо NaN
        # если для типа зданий этажности нет, удаляю
        for i in gdf['building'].unique():
            try:
                mean_level = gdf.loc[gdf['building'] == i,
                                     'building:levels'].mean()
                mean_level = int(mean_level)
                gdf.loc[gdf['building'] == i, 'building:levels'] = gdf.loc[
                    gdf['building'] == i, 'building:levels'].fillna(mean_level)
            except:
                pass
        
        if inplace:
            self.gdf['building:levels'] = gdf['building:levels']
            
        return gdf['building:levels']

    def common_area(self, reduce_factor=0.7, inplace=False):
        if inplace:
            self.gdf['common_area'] = self.area * self.complete_levels_by_type()
        return self.area * self.complete_levels_by_type() * reduce_factor
    
    def living_area(self, reduce_factor=0.9, inplace=False):
        if inplace:
            self.gdf['living_area'] = self.common_area() * reduce_factor
        return self.common_area() * reduce_factor

    # функция подсчёта количества проживающих в зданиях людей
    # принимает словарь типов зданий, для которых считается количество людей
    # подсчёт ведётся по формуле относительно общей площади здания
    # по умолчанию считает людей во всех домах
    # если дан список, только в типах домов из списка
    # подразумевается, что считаются только жилые дома, хотя ограничений нет
    # если дома к этому шагу не сгенерированы, генерирует по заданному списку
    # если списка нет, генерирует все здания, считает все здания
    # присоединяет общую площадь и людей к self.buildings
    # выводит датасет с домами с площадью и людьми
    def people(self, types=None, area_ratio=20, inplace=False):
        # выбираю нужные типы домов (жилые)
        types = self.types['accommodation'] if type(types) is not list else types
        
        gdf = self.gdf[self.gdf['building'].isin(types)]

        # перевожу геометрию в нужную црс
        gdf = gdf.to_crs(self.utm_crs)

        # считаем население каждого жилого дома, по формуле
        # площадь дома * пониж. коэффициент / м2 на человека
        # в данном случае пониж. коэффициент = 0,4, 20м2 на человека
        gdf['people'] = self.living_area() / area_ratio
        gdf.people = gdf.people.round().astype('Int64')

        # если в доме 0 чел. (слишком маленькая площадь), меняем на 1
        gdf.loc[gdf.people == 0, 'people'] = 1

        if inplace:
            self.gdf['people'] = gdf['people']
        
        return gdf['people']


class Transport(GeoFrame):
    dir_path = 'transport'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def by_query(cls, query, features_dict):
        gdf = ox.features_from_place(query, features_dict)
        
        gdf.reset_index(inplace=True)
        gdf.sort_index(inplace=True)
        
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query
        
        return gf

class Shop(GeoFrame):
    dir_path = 'transport'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def by_query(cls, query, features_dict):
        gdf = ox.features_from_place(query, features_dict)
        gdf.sort_index(inplace=True)

        gdf.reset_index(inplace=True)
        
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query
        
        return gf


class Hex(GeoFrame):
    dir_path = 'hex'
    hex_size = np.sqrt(500000 / np.sqrt(3))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # строит сетку. размер по умолчанию ~537 метров
    # это расстояние от центра гексагона до его стороны (радиус впис. окр-ти)
    # получается, что диаметр в районе километра
    # можно задать свой радиус при желании
    @classmethod
    def by_query(cls, query, size=None, *, top='flat', form='border'):
        """size - это радиус вписаной в гексагон окружности, то есть
        расстояние от центра до стороны гексагона
        по умолчанию size стоит такое, что одна ячейка по площади равна 1 км2"""

        size = cls.hex_size if size is None else size
            
        gf = super().by_query(query)

        if top not in ('flat', 'point'):
            raise ValueError("The 'top' parameter can only take the values "
                             "'flat' or 'point'. The default setting is 'flat'.")

        gdf = gf.gdf.copy()
        
        # переводим в метровую crs
        gdf = gdf.to_crs(gf.utm_crs)

        # определяем границы сетки как экстент по городу
        xmin, ymin, xmax, ymax = gdf.bounds.iloc[0]

        # создаём точки центров гексагонов
        # для этого сначала нужно создать прямоугольную сетку точек
        if top == 'flat':
            x, y = np.meshgrid(np.arange(xmin, xmax, size * np.sqrt(3)),
                               np.arange(ymin, ymax, size))
        elif top == 'point':
            x, y = np.meshgrid(np.arange(xmin, xmax, size),
                               np.arange(ymin, ymax, size * np.sqrt(3)))
        points = np.dstack((x, y))

        # чтобы сетка стала гексагональной, удаляем точки,
        # которые не "попадают" в центры гексагонов
        condition = np.sum(np.indices(points.shape[:2]), axis=0) % 2 == 0
        points = points[condition].tolist()

        # создаём вокруг точек точки вершин шестиугольников
        if top == 'flat':
            angles = [60 * i for i in range(6)]
        elif top == 'point':
            angles = [60 * i + 30 for i in range(6)]

        side_length = 2 * size / math.sqrt(3)
        hexagones = [
            [(point[0] + side_length * math.cos(math.radians(angle)),
              point[1] + side_length * math.sin(math.radians(angle)))
             for angle in angles] for point in points
        ]

        # создаём полигоны шестиугольников
        hexagones = gpd.GeoSeries([Polygon(hexagon) for hexagon in hexagones])

        # отбирает ячейки, которые попадают на границы территории
        if form=='border' :
            # выгружаю границу
            border = gf.gdf.to_crs(gf.utm_crs).loc[0, 'geometry']
            
            # выкидываю гексагоны, которые не в границах локации
            count=0
            trash = set() 
            for i in hexagones.index:
                if not shp.intersects(hexagones[i], border):
                    trash.add(i)
                count = process_bar(len(hexagones), count) 
            hexagones.drop(trash, inplace=True)
            hexagones.reset_index(drop=True, inplace=True)
        
        elif form=='box':
            pass
        
        else:
            raise AttributeError('неверный параметр form')

        # создаём геодатасекс
        gdf = gpd.GeoDataFrame(geometry=hexagones, crs=gf.utm_crs)
        
        # заношу сетку в класс, переводя в изначальную crs
        gdf = gdf.to_crs(gf.crs)
        gf = cls(gdf)
        
        # подгружаем query
        gf.query = query

        return gf


# подсчёт числа людей в каждой ячейке
def join_by_location(join_to, join_from, attr_to_join=None, inplace=False):     
    
    if attr_to_join is None:
        attr_to_join = {i:'first' for i in join_from.gdf.columns}
        del attr_to_join['geometry']
    
    # если в передаваемых аттрибутах нет геометрии, добавить
    if 'geometry' in attr_to_join:
        raise ValueError('не надо указывать geometry в attr_to_join')
    
    # копируем главный секс, сохраняем его столбцы, добавляем присоединяемые
    gdf_to = join_to.gdf.copy()
    
    cols={i:'first' for i in gdf_to.columns}
    cols = {**cols, **attr_to_join}
    
    # копирование присоединяемого секса
    gdf_from = join_from.gdf[['geometry'] + list(attr_to_join)].copy()
    
    # перевод gdf из географических в метровые координаты
    gdf_to = gdf_to.to_crs(join_to.utm_crs)
    gdf_from = gdf_from.to_crs(join_from.utm_crs)
    
    # присоединение
    gdf_to = gpd.sjoin(gdf_to, gdf_from, how='left')[list(cols)]

    # аггрегирование данных по индексу, геометрия и crs
    gdf_to = gdf_to.groupby(level=0).agg(cols)
    gdf_to = gdf_to.set_geometry('geometry')
    gdf_to.crs = join_to.utm_crs
    
    # перевод в wgs 84
    gdf_to = gdf_to.to_crs(join_to.crs)
    
    # замена исходного сета, если юзер пожелал так
    if inplace:
        join_to.gdf = gdf_to
    
    # возврат
    return gdf_to
    
def bool_by_condition(one, other, name='condition', inplace=False):
    base = one.gdf.copy()
    checking = other.gdf.copy()
    
    checking = shp.GeometryCollection(list(checking.geometry))
    
    count=0
    for i in base.index:
        if shp.intersects(base.loc[i, 'geometry'], checking):
            base.loc[i, name] = True
        count = process_bar(len(base), count) 
    
    if inplace:
        one.gdf[name] = base[name].astype('boolean')
    
    return base

def migration_cost(grid, x, grid_r_size, inplace=False):  
    
    gdf = grid.gdf.copy()
    
    # считаем время, которое потребуется на перемещение внутри каждой клетки
    # оно зависит от того, населённый ли пункт перед нами или нет, а также от наличия дорог
    # время внутри считаем как время, необходимое на преодоление радиуса клетки

    # если клетка является населённым пунктом - скорость на дороге 40 км/ч
    # клетка считается нп если в неё больше 5000 человек
    gdf.loc[gdf['people']>=5000,'inner_move'] = grid_r_size / 1000 / 40 * 60

    # если нп нет - то 60 км/ч
    gdf.loc[gdf['people']<5000,'inner_move'] = grid_r_size / 1000 / 60 * 60

    # если в клетке нет дорог (длина равна 0), то скорость 4 км/ч (человеческий шаг)
    gdf.loc[gdf['roads_len']==0,'inner_move'] = grid_r_size / 1000 / 4 * 60
      
    
    gdf['dist_min'] = float('inf')
    gdf.loc[x, 'dist_min'] = gdf.loc[x, 'inner_move']
    unvisited = gdf.dist_min.copy()
    road_time = 2 * (grid_r_size / 1000 / 90 * 60)
    
    count = 0
    cycle_len = list(unvisited.index)
    while list(unvisited.index):
        
        min_paths = min(unvisited)
        cells = unvisited[unvisited==min_paths]
        cell=min(cells.index)

        for i in gdf.neighbors[cell]:
            neigh_path = gdf.dist_min[cell] + gdf.inner_move[cell] + gdf.inner_move[i]
            if gdf.has_main_road[cell] and gdf.has_main_road[i]:
                neigh_path = gdf.dist_min[cell] + road_time
            
            if neigh_path < gdf.dist_min[i]:
                gdf.loc[i, 'dist_min'] = neigh_path
                unvisited.loc[i] = neigh_path
            
        unvisited.drop(cell, inplace=True)
        count = process_bar(len(cycle_len), count)
    
    if inplace:
        grid.gdf['dist_min'] = gdf.dist_min 
    
    return gdf.dist_min   

def check_connections(roads, grid):
    roads = roads.gdf[['geometry']].copy()
    grid = grid.gdf[grid.gdf.has_main_road][['geometry']].copy()
    
    roads = roads.assign(connects=pd.NA)
    for i in roads.index:
        inter = roads.loc[i, 'geometry'].intersects(grid)
        inter = inter[inter.geometry]
        if len(inter)>1:
            roads.at[i, 'connects'] = list(inter.index)
    
    roads = roads[roads['connects'].notna()]
    
    return roads

def check_neighbors(grid, inplace=False):
    gdf = grid.gdf.copy()
    gdf['neighbors'] = pd.NA
    count=0
    for i in gdf.index:
        n = gdf.geometry.intersects(gdf.loc[i,'geometry'].buffer(0.0000001, 1))
        n = n[n]
        n.drop(i, inplace=True)
        gdf.at[i, 'neighbors'] = list(n.index)
        count = process_bar(len(gdf), count)
    
    if inplace:
        grid.gdf = gdf
    
    return gdf
    
def surplus(grid, inplace=False):
    gdf = grid.gdf[['people', 'shop']].copy()
    
    # скольких людей может обслужить инфраструктура вообще
    gdf['capacity'] = gdf.shop * 250
    
    # бонус к производству благ каждым человеком, который даёт инфраструктура
    gdf.loc[gdf['people']>0, 'production_bonus'] = gdf.capacity / (gdf.people + gdf.capacity)
    gdf.production_bonus.fillna(0, inplace=True)
    
    # производство благ клеткой
    gdf['production'] = gdf.people * (1 + gdf.production_bonus)

    # рост потребления блага от перепроизводства
    gdf['consumpt_modif'] = 1 - (gdf.people / gdf.production)
    gdf.consumpt_modif.fillna(0, inplace=True)
    
    # общее потребление блага
    gdf['consumption'] = gdf.people * (1 + gdf.consumpt_modif)

    # излишек благ, который не смогли потребить люди в клетке
    # если людей много больше, чем инфраструктура способна обслужить,
    # фактически происходит возвращение к ситуации, будто инфраструктуры нет вовсе
    # (фактически, ситуация натурального хозяйства)
    gdf['common_profit'] = gdf.production - gdf.consumption
    
    gdf['ind_prod'] = 1 + gdf.production_bonus
    gdf['ind_cons'] = 1 + gdf.consumpt_modif
    gdf['ind_profit'] = gdf.ind_prod - gdf.ind_cons
    
    if inplace:
        grid.gdf['production'] = gdf.production
        grid.gdf['common_profit'] = gdf.common_profit
    
    return gdf['common_profit']

# задаём квери
query = 'Центральный федеральный округ'

# задаём размер ячейки
grid_r_size = Hex.hex_size * 10

# задаём границу
# border = GeoFrame.by_query(query)
# border.export(name='brd', dir_path='.')

# сетку
GRID = Hex.by_query(query, grid_r_size)
# neigbors(GRID)
# GRID.export(name='GRID', dir_path='.')

# жилые здания + считаем сколько людей живёт
BLD=Buildings.by_query(query, Buildings.types['accommodation'])
BLD.people(inplace=True)
# BLD.export(name='BLD', dir_path='.')

# считаем людей по ячейкам
join_by_location(GRID, BLD, {'people':'sum'}, inplace=True)

# выгружаем главные дороги
types = ['motorway','trunk', 'motorway_link','trunk_link']
MAIN_ROADS=Transport.by_query(query, {'highway': types})
# MAIN_ROADS.export(name='MAIN_ROADS', dir_path='.')

# # упрощаем главные дороги
MAIN_ROADS.merge_line_geometry(inplace=True)
# MAIN_ROADS.simplify(Hex.hex_size, inplace=True)

# определяем, в каких клетках есть главные дороги
join_by_location(GRID, MAIN_ROADS, {'highway':'count'}, inplace=True)
GRID.gdf['has_main_road'] = GRID.gdf['highway']>0
GRID.gdf.drop('highway', axis=1, inplace=True)

# выгружаем второстепенные дороги
types = ['primary']
primary_roads=Transport.by_query(query, {'highway': types})
types = ['secondary']
secondary_roads=Transport.by_query(query, {'highway': types})
types = ['tertiary']
tertiary_roads=Transport.by_query(query, {'highway': types})

# упрощаем второстепенные дороги
primary_roads.merge_line_geometry(inplace=True)
secondary_roads.merge_line_geometry(inplace=True)
tertiary_roads.merge_line_geometry(inplace=True)

# объединяем второстепенные дороги
MINOR_ROADS = pd.concat([primary_roads, secondary_roads, tertiary_roads],
                        ignore_index=True)
# MINOR_ROADS.export(name='MINOR_ROADS', dir_path='.')

# объединяем все дороги
ALL_ROADS = Transport(pd.concat([MINOR_ROADS.gdf[['geometry']],
                                  MAIN_ROADS.gdf[['geometry']]],
                                ignore_index=True))

# определяем длины дорог, добавляем к сетке
ALL_ROADS.gdf['roads_len'] = ALL_ROADS.length
join_by_location(GRID, ALL_ROADS, {'roads_len':'sum'}, inplace=True)

# магазины, выгружаем
COMMERCE = Shop.by_query(query, {'shop':True})
COMMERCE.gdf.geometry = COMMERCE.gdf.geometry.centroid
COMMERCE.export(name='COMMERCE', dir_path='.')

# магазины считаем
join_by_location(GRID, COMMERCE, {'shop':'count'}, inplace=True)


# =============================================================================
# АНАЛИЗ НА ОСНОВЕ ПОКАЗАТЕЛЕЙ
# =============================================================================

check_neighbors(GRID, inplace=True)

# считаем производство и потребление
surplus(GRID, inplace=True)

GRID.gdf['pop_growth'] = 0
GRID.gdf['pop_outcome'] = 0
count=0

# анализируем возможные пункты назначения, смотрим куда отправилось население
for i in GRID.gdf[GRID.gdf.people>0].index:
    # считаем время миграции относительно клетки i
    migration_cost(GRID, i, grid_r_size, inplace=True)
    
    # сохраняю клетку отдельно
    cell = GRID.gdf.loc[[i]].copy()
    
    # смотрим излишек на человека в ячейке
    ind_prof = cell.common_profit[i] / cell.people[i]
    
    # оставляем только клетки, в которые хотят мигрировать из-за большего излишка
    gdf = GRID.gdf.loc[GRID.gdf.common_profit>cell.common_profit[i]].copy()
    
    # показатель излишка с учётом дороги до него
    gdf['attr'] = (cell.dist_min[i]**2 / gdf.dist_min**2) * gdf.common_profit
    
    # отсеиваем клетки, у которых излишек с учётом дороги оказался меньше текущего излишка клетки
    gdf = gdf[gdf.attr>cell.common_profit[i]]
    
    # показатель привлекательности эмиграции в целом
    emigrate_rate = gdf.attr.sum()
    emigrate_rate /= cell.common_profit[i]
    
    if emigrate_rate==float('inf'):
        part = 0.2
    else:
        # доля уезжающих
        part = emigrate_rate/(emigrate_rate + 1) * 0.2
    part = round(part * cell.people[i])
    
    gdf['score'] = 0
    attr = gdf.attr.copy()
    for index in range(part):
        max_value = attr.idxmax()
        gdf.loc[max_value, 'score'] += 1
        attr[max_value] /= gdf.loc[max_value, 'score'] + 1
    
    GRID.gdf.loc[gdf.index,'pop_growth'] += gdf['score']
    GRID.gdf.loc[i, 'pop_outcome'] += gdf['score'].sum()
    
    print('\n')
    count = process_bar(len(GRID.gdf[GRID.gdf.people>0]), count)
   
# получаем итоговую таблицу после первой итерации
GRID.export(name='GRID', dir_path='.')



