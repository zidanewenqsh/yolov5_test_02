# ！/usr/bin/python
#######################
# version2
# version3 func
#######################

import os
import socket
import struct
import sys
import pickle



e1_path = "D:/AI-046/Projects/dents01/dents01codes"
e2_path = "D:/AI-046/Projects/woods02/woods02codes"
e3_path = "D:/AI-046/Projects/woods03/woods03codes"
e4_path = "D:/AI-046/Projects/dents04/dents04codes"
e5_path = "D:/AI-046/Projects/woods05/woods05codes"

def sendfile():
    # epath = r"" # 默认路径
    epathlist = [e1_path,e2_path,e3_path,e4_path,e5_path]
    # epathlist = [r"D:\AI-046\Projects\dents04\originaldatas"]
    inextlist = ["py"]
    exextlist = []
    excludedir = []

    argc = len(sys.argv)
    # 获取计算机名称
    hostname = socket.gethostname()
    # 获取本机ip
    # ip_addr = socket.gethostbyname(hostname)
    ip_addr = socket.gethostbyname(hostname)
    ip_port = 8080
    bufsize = 4096000
    id_ = 0x1234
    start = 2
    if argc > 1:
        ip_addr = sys.argv[1]
    if argc > 2:
        ip_port = int(sys.argv[2])
    if argc > 3:
        id = int(sys.argv[3])
    if argc > 4:
        start = int(sys.argv[4])
    if argc > 5:
        bufsize = int(sys.argv[5])
    if argc > 6:
        epath = sys.argv[6]
        epathlist.append(epath)

    # paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize, "epath": epath}
    listdict = {"epath": epathlist, "inext": inextlist, "exext": exextlist, "exdir": excludedir}

    paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize}

    paramfile = "paramdict.tmp"
    listfile = "listdict.tmp"



    # if len(epath) > 0:
    #     epathlist.append(epath)
    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k == "id":
            print(f"{i + 1}: {param_k:8s}({hex(param_v)})")
        else:
            print(f"{i + 1}: {param_k:8s}({param_v})")

    print("6: add\n7: rm\n8: load\n9: dump")
    print(listdict)
    while 1:
        try:
            chs = input("input a choice：")
            chs = int(chs)
            if chs == 1:
                vas = input("input ip_addr: ")
                paramdict["ip_addr"] = vas
            elif chs == 2:
                vas = input("input ip_port: ")
                paramdict["ip_port"] = int(vas)
            elif chs == 3:
                vas = input("input id_: ")
                paramdict['id'] = eval(f"0x{vas}")
            elif chs == 4:
                vas = input("input start: ")
                paramdict["start"] = int(vas)
            elif chs == 5:
                vas = input("input bufsize: ")
                paramdict["bufsize"] = int(vas)
            elif chs == 6:
                ld_k = list(listdict.keys())
                print(ld_k)
                chs2 = input(f"input listindex to add <{len(listdict)}: ")

                ld_v = listdict[ld_k[int(chs2)]]

                while 1:
                    # print("input a digit to break")
                    vas = input("input value, digit to break: ")
                    if vas.isdigit():
                        break
                    if ld_k[int(chs2)] == "epath":
                        if os.path.exists(vas):
                            # epathlist.append(vas)
                            ld_v.append(vas)
                            print(f"epath:{os.path.abspath(vas)}")
                            print(f"epathlist: {ld_v}")
                        else:
                            print("path not exist")
                            continue
                    else:
                        ld_v.append(vas)
                print(listdict)

            elif chs == 7:
                ld_k = list(listdict.keys())
                print(ld_k)
                chs2 = input(f"input listindex to rm <{len(listdict)}: ")

                ld_v = listdict[ld_k[int(chs2)]]
                if len(ld_v) == 0:
                    print("nothing to remove")
                    continue

                while 1:
                    print(ld_v)
                    vas = input(f"input rm index(int), -1 to break<{len(ld_v)}: ")
                    vas = int(vas)
                    if vas < 0:
                        break
                    ld_v.pop(vas)

                print(listdict)
            elif chs == 8:
                if os.path.exists(paramfile):
                    with open(paramfile, "rb") as f:
                        paramdict = pickle.load(f)
                else:
                    print("paramdict file not exist")

                if os.path.exists(listfile):
                    with open(listfile, "rb") as f:
                        listdict = pickle.load(f)
                else:
                    print("listdict file not exist")
                # print(paramdict)
                for i, (param_k, param_v) in enumerate(paramdict.items()):
                    if param_k == "id":
                        print(f"{param_k}: {hex(param_v)} ", end='')
                    else:
                        print(f"{param_k}: {param_v} ", end='')
                print()
                print(listdict)
            elif chs == 9:
                with open(paramfile, "wb") as f:
                    pickle.dump(paramdict, f)
                with open(listfile, "wb") as f:
                    pickle.dump(listdict, f)
                # print(paramdict)
                for i, (param_k, param_v) in enumerate(paramdict.items()):
                    if param_k == "id":
                        print(f"{param_k}: {hex(param_v)} ", end='')
                    else:
                        print(f"{param_k}: {param_v} ", end='')
                print()
                print(listdict)
            elif chs == -1:
                exit(0)
            else:
                break
        except Exception as e:
            print(e)
            print("input error")
            continue

    # paramdict = {"ip_addr": ip_addr, "ip_port": ip_port, "id": id_, "start": start, "bufsize": bufsize}
    ip_addr = paramdict['ip_addr']
    ip_port = paramdict['ip_port']
    id_ = paramdict['id']
    start = paramdict['start']
    bufsize = paramdict['bufsize']

    for i, (param_k, param_v) in enumerate(paramdict.items()):
        if param_k == "id":
            print(f"{param_k}: {hex(param_v)} ", end='')
        else:
            print(f"{param_k}: {param_v} ", end='')
    print()


    if len(epathlist) == 0:
        epathlist.append('./')
    print(listdict)
    # print(inextlist)
    # assert len(epathlist) > 0
    # exit()
    epathlist = listdict["epath"]
    inextlist = listdict["inext"]
    exextlist = listdict["exext"]
    excludedir = listdict["exdir"]
    # print(inextlist)
    # # assert len(epathlist) > 0
    # exit()

    SEND_BUF_SIZE = bufsize  # if bufsize != None else 40960000
    # SEND_BUF_SIZE = 4096
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # print("ip:", ip_addr, ip_port)
    p.connect((ip_addr, ip_port))

    for epath in epathlist:
        if not os.path.exists(epath):
            print(f"path {epath} not exits")
            continue
        epath = os.path.abspath(epath)
        epath = epath.replace('\\', '/')
        epath_len = len(epath.split('/'))
        start = epath_len - 1 if start == None else start
        for roots, dirs, files in os.walk(epath):
            roots = roots.replace('\\', '/')
            rootslist = roots.split('/')
            rootslist_len = len(rootslist)
            rel_len = rootslist_len - epath_len
            # print("rel_len", rel_len)
            dir_1 = rootslist[-1] if rel_len > 0 else None
            dir_2 = rootslist[-2] if rel_len > 1 else None
            dir_3 = rootslist[-3] if rel_len > 2 else None
            # print(dir_1, dir_2, dir_3)
            if dir_1 in excludedir or dir_2 in excludedir or dir_3 in excludedir:
                continue
            for file in files:
                ext = file.split('.')[-1]
                # print("ext:",ext,ext in inextlist, len(inextlist))
                if ext in exextlist or ext == "tmp":
                    continue
                if ext in inextlist or len(inextlist) == 0:

                    filepath = os.path.join(roots, file).replace("\\", '/')

                    pathlist = filepath.split('/')
                    rel_path = '/'.join(pathlist[start:]).encode('utf-8')
                    relpath_len = len(rel_path)
                    with open(filepath, 'rb') as f:
                        buffer = f.read()
                        buffer_size = len(buffer)
                        mtime = int(os.stat(filepath).st_mtime)
                        # 发送id
                        fmt = f">H"
                        msg = struct.pack(fmt, id_)
                        p.sendall(msg)
                        # 接受服务器反馈
                        flagsize = struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]

                        if flag == 0:
                            print("id error")
                            exit(-1)
                        # print("flag", flag)

                        # 发送一些基本信息，修改时间，路径长度，文件长度，最大发送长度
                        fmt = f">4I"
                        msg = struct.pack(fmt, mtime, relpath_len, buffer_size, SEND_BUF_SIZE)
                        p.sendall(msg)

                        # 发送路径
                        fmt = f">{relpath_len}s"
                        msg = struct.pack(fmt, rel_path)
                        p.sendall(msg)
                        # print("msg2", len(msg))
                        # 接收服务器的flag信息，是否发送文件
                        flagsize = struct.calcsize(">I")
                        recv_msg = p.recv(flagsize)
                        flag = struct.unpack(">I", recv_msg)[0]
                        # print("flag", flag)
                        # if flag == -1:
                        #     continue
                        # 发送
                        if flag:
                            msg = struct.pack(f">{buffer_size}s", buffer)
                            # print("msg3", len(msg))
                            if buffer_size > SEND_BUF_SIZE:
                                send_index = 0

                                while send_index < buffer_size:
                                    send_size = min(SEND_BUF_SIZE, buffer_size - send_index)
                                    p.sendall(msg[send_index:send_index + send_size])
                                    send_index += send_size
                            else:
                                p.sendall(msg)
                            print(f"file send: {filepath}")
                        else:
                            print(f"     pass: {filepath}")

    p.close()
    os.system("pause")


if __name__ == '__main__':
    sendfile()
