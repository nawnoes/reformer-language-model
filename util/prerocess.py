import os
from tqdm import tqdm

if __name__ =='__main__':
  data_dir = '../data/novel'
  out_dir = '../data/preprocessed'

  # 파일 리스트
  file_list = os.listdir(data_dir)

  # 목록 내에 .txt 파일 읽기
  # progress_bar = tqdm(file_list, position=1, leave=True)

  for file_name in file_list:
      if ".txt"in file_name :
          # progress_bar.set_description(f'file name: {file_name}')
          full_file_path = f'{data_dir}/{file_name}'
          out_file_path =  f'{out_dir}/{file_name}'
          try:
            read_f = open(full_file_path,'r',encoding='cp949')
            write_f = open(out_file_path,'w',encoding='utf-8')

            while True:
              line = read_f.readline()
              if not line: break
              write_f.write(line)


            read_f.close()
            write_f.close()
          except:
            continue




