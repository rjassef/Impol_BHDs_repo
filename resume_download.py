import subprocess
import sys
import re

###

def file_name_decode(line):

    #Get the core part of the filename. 
    m = re.match("^.*/(.*?)$",line[:-1])
    core_fname = m.group(1)

    if re.search("readme",line):
        fname = "readme_"+core_fname+".txt"
    elif re.search("calibrationxml",line):
        fname = core_fname
    elif re.search("file",line):
        if line[-4:-1]==".NL":
            fname = core_fname+".txt"
        else:
            fname = core_fname+".fits.Z"

    return fname



###

#Check script was called with the correct number of arguments. 
if len(sys.argv)!=2:
    print("Correct use: python {} download_script.sh".format(sys.argv[0]))
    exit()

#First, make a list of all the files that have been downloaded already. 
ls_output = subprocess.run("ls -lrth", shell=True, capture_output=True)
fnames = list()
for fname in ls_output.stdout.decode('utf').split()[2+8::9]:
    if fname[-3:]==".py" or fname[-3:]==".sh":
        continue
    fnames.append(fname)

#Remove the last file because it is probably not fully downloaded.
fnames.pop(-1)

#Now, recreate the download list removing the files already downloaded. 
cato = open("download_resume.sh","w")
cat = open(sys.argv[1])
for line in cat:

    #Copy the headers.
    if line[:5]!="https":
        cato.write(line) 
        continue

    #Get the file name as it would appear after download. 
    fname_ds = file_name_decode(line)

    #If the file is in the list of already downloaded files, skip it.
    if fname_ds in fnames:
        continue

    #Otherwise, set it to be downloaded. 
    cato.write(line)

cat.close()
cato.close()
