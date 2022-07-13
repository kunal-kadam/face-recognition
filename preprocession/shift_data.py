#code to restructure evaluation dataset
import os, shutil
import pandas as pd


input_path='C:/Users/Kunal Kadam/Desktop/Training_Evaluation_Dataset/Training Dataset'
output_path='C:/Users/Kunal Kadam/Desktop/Intel/data/nthu8/train'
label_type='/train/lauging/label/'
video_type = '/sleepy/'

# # files = [f for f in os.listdir(output_path+video_type) if os.path.isfile(f)]

# # print(files)
df_list = []

for i in os.listdir(output_path+video_type):
    if(i.endswith('.avi')):
        print((i, i.split('.')[0]+'_eye.txt'))
        df_list.append((i, i.split('.')[0]+'_eye.txt'))

# print(df_list)
df = pd.DataFrame(df_list, columns = ['video', 'label'])
df.to_csv('./data/nthu8/train/sleepy/sleep.csv')
# for subdir, dirs, files in os.walk(output_path+video_type):
    # for file in files:
        # print(file)
output_type = '/sleepy'

# if not os.path.exists(output_path+output_type+'/label'):
#     os.makedirs(output_path+output_type+'/label')

# for i in os.listdir(input_path):
#     for j in os.listdir(input_path+'/'+i):
#         for k in os.listdir(input_path+'/'+i+'/'+j):
#             print(k)
#             if k.split('_')[1] == 'slowBlinkWithNodding' and k.split('_')[2] == 'eye.txt':
#                 shutil.copy(input_path+'/'+i+'/'+j+'/'+k , output_path+output_type+'/label')
#                 print(j+'_'+k)
#                 os.rename(output_path+output_type+'/label/'+k, output_path+output_type+'/label/'+ j+'_'+k)
        
    
# output_type = '/lauging'

if not os.path.exists(output_path+'/lauging'):
    os.makedirs(output_path+'/lauging')
if not os.path.exists(output_path+'/sleepy'):
    os.makedirs(output_path+'/sleepy')

# if not os.path.exists(output_path+output_type+'/night_noglasses'):
#     os.makedirs(output_path+output_type+'/night_noglasses')
# if not os.path.exists(output_path+output_type+'/noglasses'):
#     os.makedirs(output_path+output_type+'/noglasses')
# if not os.path.exists(output_path+output_type+'/sunglasses'):
#     os.makedirs(output_path+output_type+'/sunglasses')

# for i in os.listdir(output_path):
#     if i in ['yawning', 'sleepy', 'lauging']: continue
#     for j in os.listdir(output_path+'/'+i):
#         if(j.split('_')[1] == 'sleepyCombination.avi'or j.split('_')[1] == 'slowBlinkWithNodding.avi' ):
#             print(i+'_'+j)
#             os.rename(output_path+'/'+i+'/'+j, output_path+'/'+i+'/'+i+'_'+j)
#             shutil.copy(output_path+'/'+i+'/'+i+'_'+j, output_path+'/sleepy')


# for j in os.listdir(f'{input_path}'):
#     if j.endswith('.mp4'):
#         if(j.split('_')[1] == 'glasses'):
#             shutil.move(f'{input_path}/{j}', f'{output_path}/{output_type}/glasses/')
#             print(f'{input_path}/{j}')
#         if(j.split('_')[1] == 'noglasses'):
#             shutil.move(f'{input_path}/{j}', f'{output_path}/{output_type}/noglasses/')
#             print(f'{input_path}/{j}')
#         if(j.split('_')[1] == 'nightnoglasses'):
#             shutil.move(f'{input_path}/{j}', f'{output_path}/{output_type}/night_noglasses/')
#             print(f'{input_path}/{j}')
#         if(j.split('_')[1] == 'nightglasses'):
#             shutil.move(f'{input_path}/{j}', f'{output_path}/{output_type}/nightglasses/')
#             print(f'{input_path}/{j}')
#         if(j.split('_')[1] == 'sunglasses'):
#             shutil.move(f'{input_path}/{j}', f'{output_path}/{output_type}/sunglasses/')
#             print(f'{input_path}/{j}')
# #                 os.rename(f'{input_path}/{i}/{j}/{k}', f'{input_path}/{i}/{j}/{i}_{k}')

