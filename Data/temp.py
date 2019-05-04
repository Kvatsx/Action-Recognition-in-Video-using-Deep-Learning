# with open('trainlist02.txt.org') as f:
#     lines = f.readlines()
#     with open('trainlist01.txt', 'a') as wf:
#         for line in lines:
#             if 'g01' in line:
#                 wf.write(line)

with open('testlist01.txt') as f:
    content = f.readlines()
    content = [x.strip('\r\n') for x in content]
f.close()
dic={}
for line in content:
    #print line
    video = line.split('/',1)[1].split(' ',1)[0]
    print(video)
    print(video.split('_',1))
    print(video.split('_',1)[1].split('.',1))
    key = video.split('_',1)[1].split('.',1)[0]
    print(key)
    # label = action_label[line.split('/')[0]] 
    # print(label)
    # dic[key] = int(label)
    #print key,label
print(dic)