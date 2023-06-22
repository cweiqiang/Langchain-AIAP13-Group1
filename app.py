"""
To run this streamlit app, run the following command in the terminal:

streamlit run app.py
"""
import streamlit as st
import pandas as pd
import folium
import requests
from streamlit_folium import folium_static


@st.cache_data
def load_data():
    df = pd.read_csv('./data/Angel-Hsu-SG-Chicken-Rice-dataset.csv')
    df.dropna(inplace=True)

    df.rename(columns={
        'E.coli count (CFU/g)\n(>490000 is replaced by 490001, <10 is replaced by 9 for sorting)': 'E.coli count (CFU/g)'}, inplace=True)
    df.rename(
        columns={'Rating (10/10) (Angel\'s rating)': 'Rating'}, inplace=True)
    df.rename(
        columns={'Stall (Review >10 as of 2023 March 21)': 'Stall'}, inplace=True)

    df['latitude'] = None
    df['longitude'] = None

    for index, row in df.iterrows():
        postal_code = row['Postal code']
        latitude, longitude = get_geocoordinates(postal_code)
        df.at[index, 'latitude'] = latitude
        df.at[index, 'longitude'] = longitude

    df.dropna(inplace=True)

    df['E.coli count (CFU/g)'] = df['E.coli count (CFU/g)'].str.replace(',',
                                                                        '').astype(int)

    return df


@st.cache_data
def get_geocoordinates(postal_code):
    url = f"https://developers.onemap.sg/commonapi/search?searchVal={postal_code}&returnGeom=Y&getAddrDetails=N"
    response = requests.get(url)
    data = response.json()

    if 'results' in data and len(data['results']) > 0:
        latitude = data['results'][0]['LATITUDE']
        longitude = data['results'][0]['LONGITUDE']
        return latitude, longitude
    else:
        return None, None


def visualize_map(df):
    singapore_map = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

    for index, row in df.iterrows():
        # Determine the color of the marker based on E.coli count
        if row['E.coli count (CFU/g)'] < 100:
            marker_color = 'green'
        elif 100 <= row['E.coli count (CFU/g)'] < 1000:
            marker_color = 'orange'
        else:
            marker_color = 'red'

        marker = folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Coordinates: {row['latitude']}, {row['longitude']}",
            icon=folium.Icon(color=marker_color)  # Use the determined color
        )

        tooltip_text = f"E.coli count (CFU/g): {row['E.coli count (CFU/g)']}<br>Cost: {row['Cost']}<br>Rating: {row['Rating']}<br>Stall: {row['Stall']}<br>Address: {row['Address']}"
        tooltip = folium.Tooltip(tooltip_text)
        marker.add_child(tooltip)
        marker.add_to(singapore_map)

    return singapore_map


def main():
    st.title("Singapore Chicken Rice Map")

    # Load the data
    df = load_data()

    # Visualize the map
    singapore_map = visualize_map(df)

    # Render the map using Streamlit
    _ = st.markdown(folium_static(singapore_map), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
