from ftplib import FTP
from elechons import config
import tarfile
import os

def acorn_ftp_download(filename):
    ftp = FTP(config.FTP_SERVER)
    ftp.login()
    ftp.cwd(config.FTP_DIR)

    tarname = filename + '.tar.gz'

    tarpath = os.path.join(config.RAW_DATA_DIR, tarname)
    folderpath = os.path.join(config.RAW_DATA_DIR, filename)

    print(f'downloading {tarname}')
    with open(tarpath, 'wb') as f:
        ftp.retrbinary(f'RETR {tarname}', f.write)
    
    print(f'extracting {tarname} into elechons/data/raw/{filename}')
    with tarfile.open(tarpath, 'r:gz') as tar:
        tar.extractall(path=folderpath, filter='data')

    print(f'deleting {tarname}')
    os.remove(tarpath)

    ftp.quit()

def download():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    for stat in config.STAT_TYPES.keys():
        acorn_ftp_download(config.ACORN_PREFIX + stat)
    
    acorn_ftp_download(config.STATION_LOCATION_FILE)
    
    print('all datasets downloaded and extracted into elechons/data/raw/')

if __name__ == '__main__':
    download()
