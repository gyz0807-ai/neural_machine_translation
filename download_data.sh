mkdir -p ./dataset
wget -O ./dataset/cmn-eng.zip "http://www.manythings.org/anki/cmn-eng.zip"
unzip ./dataset/cmn-eng.zip -d ./dataset/
rm -rf ./dataset/cmn-eng.zip