# Prova Gpu
Il codice in get_volume_3d.py prende in input i pesi della rete e un volume tif o dcm e crea un volume in output con formato .dcm

## Come usare

python3 get_volume_3d.py [pesi_rete] [input] [output_name]

- pesi_rete => .pth.tar
- input => .tif .tiff .dcm
- output_name => nome volume di output

## Esempio
`python3 get_volume_3d.py volumeHD_AverageMetal_subsample2x.dcm out_volume`


