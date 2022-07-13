import os
from tqdm import tqdm
import socket

IP = socket.gethostbyname(socket.gethostname())
PORT = 4456
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"


def send_file_v2(file_path):
    """ TCP socket and connecting to the server. """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    client.send("SEND".encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)

    """ Sending the filename and filesize to the server. """
    FILEPATH = file_path
    FILENAME = FILEPATH.split('/')[-1]
    FILESIZE = os.path.getsize(FILEPATH)
    data = f"{FILENAME}_{FILESIZE}"
    client.send(data.encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)
    print(f"SERVER: {msg}")

    """ Data transfer. """
    bar = tqdm(range(FILESIZE), f"Sending {FILENAME}", unit="B", unit_scale=True, unit_divisor=SIZE)
    with open(FILEPATH, "r") as f:
        while True:
            data = f.read(SIZE)
            if not data:
                break
            client.send(data.encode(FORMAT))
            msg = client.recv(SIZE).decode(FORMAT)

            bar.update(len(data))
    
    """ Closing the connection """
    client.close()


def receive_file_v2(filename, folder_path):
    """ TCP socket and connecting to the server. """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    client.send("RECV".encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)

    """ Sending the filename and filesize to the server. """
    FILENAME = filename
    client.send(FILENAME.encode(FORMAT))
    data = client.recv(SIZE).decode(FORMAT)
    item = data.split("/")
    FILENAME = item[0]
    FILESIZE = int(item[1])


    """ Receiving Data. """
    bar = tqdm(range(FILESIZE), f"Receiving {FILENAME}", unit="B", unit_scale=True, unit_divisor=SIZE)
    with open(f"{folder_path}/{FILENAME}", "w") as f:
        while True:
            data = client.recv(SIZE).decode(FORMAT)
            if not data:
                break
            f.write(data)
            client.send("Data received.".encode(FORMAT))

            bar.update(len(data))

    """ Closing the connection """
    client.close()

    return f"{folder_path}/{FILENAME}"
