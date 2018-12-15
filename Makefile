execute :
	python3 code/main.py

init :
	mkdir -p model
	mkdir -p result
	
	wget -c https://maups.github.io/tcv3/data_part1.tar.bz2
	tar -jxvf data_part1.tar.bz2
	cp -rf ./data_part1/* ./data/
	rm -rf ./data_part1*