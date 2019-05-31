mkdir -p ./dataset
wget -O ./dataset/en-zh.tmx.gz "http://opus.nlpl.eu/download.php?f=News-Commentary/v11/tmx/en-zh.tmx.gz"
gzip -d ./dataset/en-zh.tmx.gz