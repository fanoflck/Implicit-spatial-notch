cd CNNDetection

for i in stylegan deepfake gaugan cyclegan whichfaceisreal crn seeingdark stylegan2 stargan imle san biggan
do 
    echo "当前的数据是${i}"
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.1.pth -c 224 -n gau 
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.1.pth -c 224 -n uni
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.1.pth -c 224 -n gau -f
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.1.pth -c 224 -n uni -f
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.5.pth -c 224 -n gau 
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.5.pth -c 224 -n uni 
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.5.pth -c 224 -n gau -f
    python demo_dir.py -d dataset/test/$i -m weights/blur_jpg_prob0.5.pth -c 224 -n uni -f
done