import pandas as pd

# import the library
import folium
import webbrowser

# Make a data frame with dots to show on the map
data = pd.DataFrame({
   'lon':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
   'lat':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
   'name':['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador'],
   'value':[10, 12, 40, 70, 23, 43, 100, 43]
}, dtype=str)

# Make an empty map
n = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)

# add marker one by one on the map
for i in range(0,len(data)):
    html = f"""
           <h1> {data.iloc[i]['name']}</h1>
           <p>You can use any html here! Let's do a list:</p>
           <ul>
               <li>Item 1</li>
               <li>Item 2</li>
           </ul>
           </p>
           <p>And that's a <a href="https://www.python-graph-gallery.com">link</a></p>
           """
    iframe = folium.IFrame(html=html, width=200, height=200)
    folium.Marker(
        location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
        popup=data.iloc[i]['name'],
        icon=folium.DivIcon(html=f"""
        <div style="font-family: courier new; color: blue">{data.iloc[i]['name']}<svg>
                <rect x="0", y="0" width="10" height="10", fill="red", opacity=".3">
            </svg></div>""")
    ).add_to(n)

n.save('1map.html')
webbrowser.open_new_tab('1map.html')