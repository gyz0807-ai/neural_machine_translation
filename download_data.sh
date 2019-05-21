wget -O ./dataset/parallel_nc_compressed.tgz "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
tar -xvzf ./dataset/parallel_nc_compressed.tgz -C ./dataset/
mv ./dataset/training/news-commentary-v12.zh-en.zh ./dataset/
mv ./dataset/training/news-commentary-v12.zh-en.en ./dataset/
rm ./dataset/parallel_nc_compressed.tgz
rm -rf ./dataset/training