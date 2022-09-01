# About dealing_with_docs

This repo contains a tool to perform two actions with a pdf volume of several pages:

- 🧹[Filtering Volume](#filtering-volume)

- ✂️ 📃📃📃[Volume Fragmentation](#volume-fragmentation)

###  Filtering Volume


This tool can be used to filter a pdf volume with several pages to obtain a reduced version of
the original pdf file. The user can choose to keep certain pages or discard them based on the following features:

- Empty pages (always discarded)
- Pages with shields

#### Context

#### Dependencies


#### Usage

###  Volume Fragmentation

This tool can be used to split a pdf volume with several pages to obtain subdocuments corresponding to single files originally integrate in the original pdf volume. 
The difference between subdocuments is detected based on background changes, size changes and shield detection. Empty pages will be discarded.

#### Context



#### Dependencies


#### Usage

