# 🧽✂️ Dealing with Docs with OpenCV 🖼️🛡️

This repo contains a tool to perform two actions with a pdf volume of several pages:

- 🧹 Filtering Volume

This tool can be used to filter a pdf volume with several pages to obtain a reduced version of
the original pdf file. The user can choose to keep certain pages or discard them based on the following features:

- Empty pages (always discarded)
- Pages with shields


- ✂️ 📃📃📃 Volume Fragmentation

This tool can be used to split a pdf volume with several pages to obtain subdocuments corresponding to single files originally integrate in the original pdf volume. 
The difference between subdocuments is detected based on background changes, size changes and shield detection. Empty pages will be always discarded.

#### Context

This tool was conceived to process scanned documents. More in particular, the shield detection comes from the fact that the documents were official
administration volumes.

The idea is to reduce size of this huge volumes containing subdocuments (action clean) or splitting in different pdf files contained in the complete volume pdf for further processing.

#### Dependencies
- poppler is included within the project as its path must be specified to be able to execute the code. 
- required libraries in requierements.txt

```console
pip install -r requirements.txt
```
#### Usage

There are two modes to execute this tool in terms of what is our starting point. The pdf volume must be in a folder inside ./input/ with name <volume_name>.

- PDF: First time executing the volume and pdf to image must be done (in order to obtain single images per volume page)
```
dealing_with_docs>python dealing_with_docs.py <volume_name> *PDF* <action>
```
- IMG: When pdf to image has already be done and this step can be skipped
```
dealing_with_docs>python dealing_with_docs.py <volume_name> *IMG* <action>
```

Also, the two tools contained in this scripts are activated with the <action> argument:

- action = clean_keep : Filtered pages will be kept
- action = clean_discard : Filtered pages will be discarded
- action = split : Volume will be fragmented


