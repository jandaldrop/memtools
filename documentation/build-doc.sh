#create a basic documentation using pydoc

source_files_to_document="""
../__init__.py
../igle.py
../igleplot.py
../flist.py
"""

cat _header.txt $source_files_to_document | grep -v __future__ | grep -v 'from .' >> memtools_documentation.py
pydoc -w memtools_documentation
rm memtools_documentation.py
