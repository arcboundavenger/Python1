import geocoder # pip install geocoder

BING_KEY = "AqO0CS624sxRrFtEhjmczen1jp3QLrJPJOEdo14V7fSZ5Dvt6QPHdxgOo6ei5P46"

g = geocoder.bing('krafton', key=BING_KEY)
# g = geocoder.google('white house', method='places')
print(g.json['address'])
