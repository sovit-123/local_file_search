"""
Minimal script to check PDF file manipulations.
"""

from pypdf import PdfReader

reader = PdfReader('../data/pdfs/2407.09025v1.pdf')
num_pages = len(reader.pages)

print('#'*50, ' NUM PAGES ', '#'*50)
print(num_pages)

page0 = reader.pages[0]
print('#'*50, ' PAGE 0 ', '#'*50)
print(page0.extract_text())

all_text = ''
for page in reader.pages:
    all_text += page.extract_text() + ' '

print('#'*50, ' ALL PAGES ', '#'*50)
print(all_text)