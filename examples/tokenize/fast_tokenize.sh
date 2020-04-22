for part in {2..7}; do
	python3 /home/orion/git/sentence-transformers/examples/tokenize/faster_tokenize_bert.py --data_dir "/media/orion/Yeni Birim/patent_data/v8/dataset/v2/desc/bert/" \
	--model_name_or_path /home/orion/patent_stuff/models/scibert_scivocab_uncased \
	--part_no $part &
done

BACK_PID=$!
wait $BACK_PID