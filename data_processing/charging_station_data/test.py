# KlZy8pfpWd1uclJene6sMZdsLRUfxJPs
import requests


def search_nearby_poi_baidu(
    lon,
    lat,
    query="特来电充电站",
    radius=1500,
    ak="KlZy8pfpWd1uclJene6sMZdsLRUfxJPs",
    max_pages=3
):
    url = "https://api.map.baidu.com/place/v3/around"

    all_results = []

    for page in range(max_pages):

        params = {
            "query": query,
            "location": f"{lat},{lon}",   # 注意：百度是 lat,lng
            "radius": radius,
            "radius_limit": "true",
            "scope": 2,
            "sort_name": "distance",      # ⭐按距离排序
            "page_num": page,
            "page_size": 20,              # 最大值
            "output": "json",
            "ak": ak
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != 0:
            print("请求失败:", data)
            break

        results = data.get("results", [])

        if not results:
            break

        all_results.extend(results)

    # 去重（按名称）
    unique_results = {}
    for poi in all_results:
        name = poi.get("name")
        if name not in unique_results:
            unique_results[name] = poi

    final_results = list(unique_results.values())

    # 输出
    for poi in final_results:
        print("Name:", poi.get("name"))
        print("Address:", poi.get("address"))
        print("longitudes and latitude:", poi.get("location"))
        print("-" * 30)

    print(f"======================================")
    print(f"总共获取到 {len(final_results)} 个POI")

    return final_results

def search_nearby_poi_amap(
    lon,
    lat,
    query="特来电充电站",
    radius=1000,
    key="ca1654b5b9e0a4cd6029fb19959e94df",
    max_pages=3
):
    url = "https://restapi.amap.com/v5/place/around"

    all_results = []

    for page in range(1, max_pages + 1):

        params = {
            "key": key,
            "keywords": query,
            "location": f"{lon},{lat}",   # 高德是 lon,lat
            "radius": radius,
            "page_num": page,
            "page_size": 25,
            "output": "json"
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "1":
            print("请求失败:", data)
            break

        pois = data.get("pois", [])

        if not pois:
            break

        all_results.extend(pois)

    # 去重：优先按 POI id 去重
    unique_results = {}
    for poi in all_results:
        poi_id = poi.get("id") or poi.get("name")
        if poi_id not in unique_results:
            unique_results[poi_id] = poi

    final_results = list(unique_results.values())

    # 按距离排序
    final_results.sort(
        key=lambda x: int(x.get("distance", 999999))
        if str(x.get("distance", "")).isdigit()
        else 999999
    )

    for poi in final_results:
        print("名称:", poi.get("name"))
        print("地址:", poi.get("address"))
        print("坐标:", poi.get("location"))
        print("距离:", poi.get("distance"), "米")
        print(poi)
        print("-" * 30)

    print("======================================")
    print(f"总共获取到 {len(final_results)} 个POI")

    return final_results

def parse(strs):
    lon, lat = strs.split(",")
    return float(lon), float(lat)

if __name__ == '__main__':
    lon_lat = [
        "116.347697,39.783707",
        # "116.437854,39.794986", "116.510955,39.826464", "116.555424,39.875297", "116.553597,39.900170", "116.550551,39.916121",
        # "116.356472,39.903931", "116.292430,40.021066", "116.506831,40.006078", "116.217250,39.932519", "116.361112,39.973950", "116.355544,39.874615",
        # "116.231172,39.852442", "116.636098,39.802269", "116.490722,40.162926", "116.716111,39.930799", "116.336330,39.706582", "116.108686,39.710934"
    ]
    for string in lon_lat:
        lon, lat = parse(string)
        search_nearby_poi_baidu(lon, lat)