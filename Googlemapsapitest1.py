import googlemaps

gmaps = googlemaps.Client(key='AIzaSyD7WhqVqldSHj6a5c3q5FN9nY_scoJasew')

search_loction = gmaps.places(('40.714224, -73.961452'), type="movie_theater")
print (search_loction)