import matplotlib.pyplot as plt

map_iou = {'iteration': [], 'AP': [], 'AP50': [], 'AP75': []}
map_giou = {'iteration': [], 'AP': [], 'AP50': [], 'AP75': []}

best_iou = {'iteration_50': -1, 'best_50': 0,
            'iteration_75': -1, 'best_75': 0, 
            'iteration_ap': -1, 'best_ap': 0}
best_giou = {'iteration_50': -1, 'best_50': 0, 
             'iteration_75': -1, 'best_75': 0, 
             'iteration_ap': -1, 'best_ap': 0}

with open('map.txt', 'r') as f:
    map_results = f.readlines()

for map_result in map_results:
    map_result = map_result.strip('\n')
    data = map_result.split(',')
    iteration = int(data[1])
    AP = float(data[2])
    AP50 = float(data[3])
    AP75 = float(data[8])
    if data[0] == 'iou':
        if AP50 > best_iou['best_50']:
            best_iou['best_50'] = AP50
            best_iou['iteration_50'] = iteration
        if AP75 > best_iou['best_75']:
            best_iou['best_75'] = AP75
            best_iou['iteration_75'] = iteration
        if AP > best_iou['best_ap']:
            best_iou['best_ap'] = AP
            best_iou['iteration_ap'] = iteration
        map_iou['iteration'].append(iteration)
        map_iou['AP'].append(AP)
        map_iou['AP50'].append(AP50)
        map_iou['AP75'].append(AP75)
    elif data[0] == 'giou':
        if AP50 > best_giou['best_50']:
            best_giou['best_50'] = AP50
            best_giou['iteration_50'] = iteration
        if AP75 > best_giou['best_75']:
            best_giou['best_75'] = AP75
            best_giou['iteration_75'] = iteration
        if AP > best_giou['best_ap']:
            best_giou['best_ap'] = AP
            best_giou['iteration_ap'] = iteration
        map_giou['iteration'].append(iteration)
        map_giou['AP'].append(AP)
        map_giou['AP50'].append(AP50)
        map_giou['AP75'].append(AP75)

print('Best AP50 IOU is {}, at {} iteration, Best AP75 IOU is {}, at {} iteration, Best AP IOU is {}, at {} iteration'
      .format(best_iou['best_50'], best_iou['iteration_50'],
              best_iou['best_75'], best_iou['iteration_75'],
              best_iou['best_ap'], best_iou['iteration_ap']))

print('Best AP50 gIOU is {}, at {} iteration, Best AP75 gIOU is {}, at {} iteration, Best AP gIOU is {}, at {} iteration'
      .format(best_giou['best_50'], best_giou['iteration_50'],
              best_giou['best_75'], best_giou['iteration_75'],
              best_giou['best_ap'], best_giou['iteration_ap']))

plt.figure(figsize=(10, 8), dpi=200)
plt.xlim((0, max(map_iou['iteration'])))
plt.ylim((0, 1))
plt.xlabel('Iteration')
plt.ylabel('mAP')
plt.plot(map_iou['iteration'], map_iou['AP'], color='b', linewidth=2.0, label='AP')
plt.plot(map_iou['iteration'], map_iou['AP50'], color='r', linewidth=2.0, label='AP50')
plt.plot(map_iou['iteration'], map_iou['AP75'], color='g', linewidth=2.0, label='AP75')
plt.legend(loc='upper right')
plt.savefig('iou.png')
plt.show()

plt.figure(figsize=(10, 8), dpi=200)
plt.xlim((0, max(map_giou['iteration'])))
plt.ylim((0, 1))
plt.xlabel('Iteration')
plt.ylabel('mAP')
plt.plot(map_giou['iteration'], map_iou['AP'], color='b', linewidth=2.0, label='AP')
plt.plot(map_giou['iteration'], map_iou['AP50'], color='r', linewidth=2.0, label='AP50')
plt.plot(map_giou['iteration'], map_iou['AP75'], color='g', linewidth=2.0, label='AP75')
plt.legend(loc='upper right')
plt.savefig('giou.png')
plt.show()

