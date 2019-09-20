import shutil
import os


trainPrefix = 'train/'
testPrefix = 'test/'
for lines in open('cars_annos.txt'):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    fname = lines[0].split('/')[1]
    if classInd <= 98:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(lines[0], ddr + '/' + fname)
    else:
        ddr = testPrefix + lines[1]
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(lines[0], ddr + '/' + fname)