rm -r datasets
mkdir datasets
FILE=celeba

URL="https://d.pcs.baidu.com/file/c9dee16efae2f4a894804e22ac39d2d5?fid=2785600790-250528-435381571174739&amp;dstime=1587198604&amp;rt=sh&amp;sign=FDtAERVY-DCb740ccc5511e5e8fedcff06b081203-nylWj%2FSahsCCwiQYz9l4DdIRCVo%3D&amp;expires=1h&amp;chkv=1&amp;chkbd=0&amp;chkpc=et&amp;dp-logid=2521737744357766988&amp;dp-callid=0&amp;shareid=2609218492&amp;r=288147903"
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
