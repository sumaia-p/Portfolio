mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
[theme]\n\
base="dark"\n\
primaryColor="#a2989a"\n\
secondaryBackgroundColor="#18181a"\n\
font="serif"\n\
" > ~/.streamlit/config.toml
