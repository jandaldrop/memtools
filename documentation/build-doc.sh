#create a basic documentation using pydoc

source_files_to_document="""
../memtools/__init__.py
../memtools/igle.py
../memtools/igleplot.py
../memtools/flist.py
"""

cat _header.txt $source_files_to_document | grep -v __future__ | grep -v 'from .' >> memtools_documentation.py
pydoc -w memtools_documentation
rm memtools_documentation.py
