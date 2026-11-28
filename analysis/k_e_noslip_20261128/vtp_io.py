"""Read the inline zlib-compressed VTK XML arrays exported by PhysicsNeMo."""
import base64
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

TYPES = {'Float32':'f4', 'Float64':'f8', 'Int32':'i4', 'Int64':'i8',
         'UInt32':'u4', 'UInt64':'u8', 'UInt8':'u1'}

def read_vtp(path):
    root = ET.parse(path).getroot()
    endian = '<' if root.get('byte_order') == 'LittleEndian' else '>'
    hd = np.dtype(endian + TYPES[root.get('header_type','UInt32')])
    def decode(e):
        dtype = np.dtype(endian + TYPES[e.get('type')])
        text = ''.join((e.text or '').split())
        if e.get('format') == 'ascii':
            a = np.fromstring(e.text, sep=' ', dtype=dtype)
        elif e.get('format') == 'binary' and root.get('compressor') == 'vtkZLibDataCompressor':
            pre = base64.b64decode(text[:4*((3*hd.itemsize+2)//3)])
            n = int(np.frombuffer(pre,dtype=hd,count=1)[0])
            count = 4*(((3+n)*hd.itemsize+2)//3)
            header = np.frombuffer(base64.b64decode(text[:count]),dtype=hd)
            payload = base64.b64decode(text[count:])
            blocks=[]
            offset=0
            for length in header[3:]:
                length=int(length)
                blocks.append(zlib.decompress(payload[offset:offset+length]))
                offset+=length
            raw=b''.join(blocks)
            expected=(n-1)*int(header[1])+int(header[2]) if n else 0
            if len(raw)!=expected or offset!=len(payload):
                raise ValueError('VTK compressed length mismatch')
            a=np.frombuffer(raw,dtype=dtype).copy()
        else:
            raise ValueError('Unsupported VTK encoding: '+str(path))
        components=int(e.get('NumberOfComponents','1'))
        return a.reshape(-1,components) if components>1 else a
    piece=root.find('./PolyData/Piece')
    data={e.get('Name'):decode(e) for e in piece.findall('./PointData/DataArray')}
    xyz=decode(piece.find('./Points/DataArray'))
    data.update(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2])
    if any(len(v)!=int(piece.get('NumberOfPoints')) for v in data.values()):
        raise ValueError('VTK point count mismatch')
    return data

if __name__=='__main__':
    import json
    folder=Path(__file__).parent/'data/k_e_noslip/step_0400000'
    for p in sorted(folder.rglob('*.vtp')):
        d=read_vtp(p)
        print(p.relative_to(folder),len(d['x']))
        print(json.dumps({k:[float(np.min(v)),float(np.median(v)),float(np.max(v))] for k,v in d.items()}))
