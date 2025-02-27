import numpy as np
import argparse

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

def main():
    """Command-line interface to compute distance between two lat/long pairs."""
    parser = argparse.ArgumentParser(description='Calculate distance between two points on Earth.')
    parser.add_argument('lat1', type=float, help='Latitude of the first point (e.g., -33.8607)')
    parser.add_argument('lon1', type=float, help='Longitude of the first point (e.g., 151.2050)')
    parser.add_argument('lat2', type=float, help='Latitude of the second point (e.g., -37.8079)')
    parser.add_argument('lon2', type=float, help='Longitude of the second point (e.g., 144.9700)')
    args = parser.parse_args()

    distance = haversine(args.lat1, args.lon1, args.lat2, args.lon2)
    print(f"Distance between ({args.lat1}, {args.lon1}) and ({args.lat2}, {args.lon2}): {distance:.2f} km")

if __name__ == '__main__':
    main()
