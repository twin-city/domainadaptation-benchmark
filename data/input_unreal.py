import json
import os
import requests

print(os.getcwd())

building_path = "data/Saint Augustin/building.json"


with open(building_path) as jsonFile:
    building_json = json.load(jsonFile)


building_json["buildings"][0]["coordonnees"]

#%%

def checkUrl(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print("Success")
        return response
    # Code here will only run if the request is successful
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)

#%%

url = 'https://opendata.paris.fr/api/records/1.0/search/?dataset=quartier_paris&q=&rows=10000&facet=l_qu&facet=c_ar'
response = checkUrl(url).json()

#%%

import matplotlib.pyplot as plt

q0, q1 = 0, 6

import numpy as np
quartier0 = np.array(response["records"][q0]["fields"]["geom"]["coordinates"])[0]
quartier1 = np.array(response["records"][q1]["fields"]["geom"]["coordinates"])[0]


np.intersect1d(quartier0, quartier1)

#%% Create correspondance dict



for neighboorhood in response["records"]:
    print(int(neighboorhood["fields"]["c_qu"]), neighboorhood["fields"]["l_qu"])


#%%
vertices = np.zeros((80,80))

neighboorhing_dict = {}

for q0 in range(80):

    # Quartier 0
    quartier0 = np.array(response["records"][q0]["fields"]["geom"]["coordinates"])[0]
    quartier0_id = int(response["records"][q0]["fields"]["c_qu"])
    quartier0_name = (response["records"][q0]["fields"]["l_qu"])
    neighboorhing_dict[quartier0_id] = []


    for q1 in range(80):
        quartier1 = np.array(response["records"][q1]["fields"]["geom"]["coordinates"])[0]
        quartier1_id = int(response["records"][q1]["fields"]["c_qu"])
        quartier1_name = (response["records"][q1]["fields"]["l_qu"])

        if len(np.intersect1d(quartier0, quartier1)) > 0 and q0 != q1:
            vertices[quartier0_id-1, quartier1_id-1] = 1
            neighboorhing_dict[int(quartier0_id)].append(quartier1_id)

plt.imshow(vertices)
plt.show()

with open("data/neighboor_graph.json", 'w') as fh:
    json.dump(neighboorhing_dict, fh)



#%%

import matplotlib.pyplot as plt

q0, q1 = 0, 9

import numpy as np
quartier0 = np.array(response["records"][q0]["fields"]["geom"]["coordinates"])[0]
quartier1 = np.array(response["records"][q1]["fields"]["geom"]["coordinates"])[0]


fig, ax = plt.subplots(1,1)
ax.scatter(quartier0[:,0], quartier0[:,1], label=response["records"][q0]["fields"]["l_qu"])
ax.scatter(quartier1[:,0], quartier1[:,1], label=response["records"][q1]["fields"]["l_qu"])
plt.legend()
plt.show()

#%%

l = []

for dist in response["records"]:
    print(dist["fields"]["c_qu"])
    l.append(int(dist["fields"]["c_qu"]))

print(l.sort())

#%%

import requests


class District:
    def __init__(self):
        self.neighbors = []


def is_overlaping(aPolygon, bPolygon):
    for point in aPolygon:
        if point in bPolygon:
            return True

    return False


def generate_district_from_fields(json_fields):
    district = District()

    # Dump json fields into new Python fields
    for k in json_fields.keys():
        setattr(district, k, json_fields[k])

    return district


def generate_neighborhood(districts):
    # Open list will be filled with futur neighbors
    open_list = [districts[0]]

    # Closed list contains all districts without those already checked
    closed_list = districts.copy()

    for aDistrict in open_list:
        closed_list.remove(aDistrict)

        aCoords = aDistrict.geom['coordinates'][0]

        for bDistrict in closed_list:

            bCoords = bDistrict.geom['coordinates'][0]

            if is_overlaping(aCoords, bCoords):
                # Link the two districts together
                aDistrict.neighbors.append(bDistrict)
                bDistrict.neighbors.append(aDistrict)

                # To avoid duplicates we check if the district is not already in the list
                if bDistrict not in open_list:
                    open_list.append(bDistrict)


def generate_districts(districts_url):
    # Download discrits info
    data_districts = requests.get(districts_url)

    # If request fails
    if data_districts.status_code != 200:
        raise "hihihi ha"

        # Convert to JSON to access data
    districts_json = data_districts.json()
    records = districts_json['records']

    districts = []

    for district_record in records:
        district = generate_district_from_fields(district_record['fields'])
        districts.append(district)

    generate_neighborhood(districts)

    return districts


def select_district(id):
    return next(district for district in districts if district.c_qu == id)  # Return an entire list -> not optimized


# void main()
# rows are set to 10000, else it returns only 10 elements
url = 'https://opendata.paris.fr/api/records/1.0/search/?dataset=quartier_paris&q=&rows=10000&facet=l_qu&facet=c_ar'

districts = generate_districts(url)

district_id = input('Enter district ID\n')

selected_district = select_district(district_id)

for neighbor in selected_district.neighbors:
    print(neighbor.c_qu + ' - ' + neighbor.l_qu)