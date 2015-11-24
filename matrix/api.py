import urllib2
import xml.etree.ElementTree as ET
import re

from log import Log

WOLFRAM_API = "http://api.wolframalpha.com/v2/query?input=%s&appid=X268PE-L3KK2H3KYL&podstate=Approximate+forms"

# FUCK APIS
# TODO: Implement parser for exact forms of result to display in LaTeX

def call_api(inp, tag=""):
    Log.i("Wolfram API call: %s" % inp)

    inp = '+'.join(inp.split())
    url = WOLFRAM_API % inp
    content = None
    try:
        # Makes the API call and gets the data, read by 'content'
        content = urllib2.urlopen(url).read()
    except urllib2.URLError:
        return
    if content:
        root = ET.fromstring(content) 	# Makes an Element Tree
        values = [[res.find('plaintext').text
                   for res in root[1].findall('subpod')]]
        if tag:
            for child in root.findall('pod'):
                if child.attrib['title'] == tag:
                    values.append([res.find('plaintext').text
                                   for res in child.findall('subpod')])
                    break
        # ans = re.compile(r'^[^0-9]*([-]?\d+[.]?\d*)[^0-9]*$')
        print values
        # result = [ans.match(res).group(1) for res in values]
        result = [[res[((res.find("~~") + 1) or (res.find("=") + 1)) + 1:]
                   for res in val]
                  for val in values]

        # TODO: For parsing exact forms - Replace keywords
        # space = re.compile(r'[^,] ')
        # result = [[space.sub(lambda x: '*', res) for res in val] for val in result]

        # sqrt = re.compile(r'sqrt([(].*?[)])')
        # result = [[sqrt.sub(lambda x: r'%s ** 1/2' % x.group(1), res) for res in val] for val in result]

        Log.d("Api Call result: %s" % result)

        return result
