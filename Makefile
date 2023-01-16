ui: # hello world
	@ streamlit run ui.py > /dev/null 2>&1 &

backend: # install requirements
	@python server.py

