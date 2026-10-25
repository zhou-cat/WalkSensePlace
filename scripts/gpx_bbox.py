# scripts/gpx_bbox.py
import glob, sys, xml.etree.ElementTree as ET
lat, lon = [], []
for f in glob.glob((sys.argv[1] if len(sys.argv) > 1 else 'GPS') + '/*.gpx'):
    for pt in ET.parse(f).getroot().iter():
        if pt.tag.split('}')[-1] == 'trkpt':
            lat.append(float(pt.attrib['lat'])); lon.append(float(pt.attrib['lon']))
pad = 0.01   # about 1.1 km of margin
print(f'south,west,north,east: {min(lat)-pad:.4f},{min(lon)-pad:.4f},'
      f'{max(lat)+pad:.4f},{max(lon)+pad:.4f}')