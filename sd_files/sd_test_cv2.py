import cv2

print("[Sucess] Import cv2")

# https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo
# $ sudo su
# $ apt update

# ImportError: libSM.so.6: cannot open shared object file: No such file or directory
# $ apt install -y libsm6

# ImportError: libXrender.so.1: cannot open shared object file: No such file or directory
# $ apt install -y libxrender-dev
# $ exit

# This command can solve
# $ apt install -y libsm6 libxrender-dev

# This is additional
# $ apt install -y libxext6