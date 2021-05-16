#!/usr/bin/python
import socket
import struct
import os
import torch
import traceback
#获取计算机名称
hostname = socket.gethostname()
#获取本机ip
ip_addr = socket.gethostbyname(hostname)
# 明确配置变量
ip_port = 8080
print("ip:",ip_addr, ip_port)
back_log = 5
savedir = "./"
# buffer_size = 1024
# 创建一个TCP套接字
ser = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   # 套接字类型AF_INET, socket.SOCK_STREAM   tcp协议，基于流式的协议
ser.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)  # 对socket的配置重用ip和端口号
# 绑定端口号
ser.bind((ip_addr, ip_port))  #  写哪个ip就要运行在哪台机器上
# 设置半连接池
ser.listen(back_log)  # 最多可以连接多少个客户端
# def func():
#     print("helloworld")
SEND_BUF_SIZE = 4096 # 发送缓冲区的大小
RECV_BUF_SIZE = 4096 # 接收缓冲区的大小
while 1:
    print("start")
    # 阻塞等待，创建连接
    conn,address = ser.accept()  # 在这个位置进行等待，监听端口号
    while 1:
        try:
            # 接受套接字的大小，怎么发就怎么收
            # buffer_size = 1024 # 也可以
            fmt = f">H3I"
            buffer_size = struct.calcsize(fmt)
            # print("buffer_size", buffer_size)

            msg = conn.recv(buffer_size)
            # print("msg1", len(msg))
            if not msg:
                # 断开连接
                conn.close()
                break
            id, mtime, relpath_len, buffer_size  = struct.unpack(fmt, msg)
            # print("id:", id, flag)
            msg = conn.recv(relpath_len)
            rel_path = struct.unpack(f">{relpath_len}s", msg)[0].decode('utf-8')
            # print("msg2", len(msg))
            # print(rel_path)
            savepath = os.path.join(savedir, rel_path)

            flag = 1
            if os.path.exists(savepath):
                mtime_ = int(os.stat(savepath).st_mtime)
                # file_size = os.path.getsize(savepath)
                # print("filesize",file_size,buffer_size, file_size==buffer_size)
                # time.strftime("%b/%d/%Y %H:%M:%S", time.localtime())
                # print("time", mtime_, mtime, mtime_>mtime)
                if mtime_ >= mtime:
                    print(f"passed file: {savepath}")
                    # print(f"passed file: {savepath:<50s} oldfile_time: {mtime_}, sendfile_time: {mtime}")
                    flag = 0
            flag_msg = struct.pack(">I", flag)
            conn.send(flag_msg)
            if flag:
                # print("flag", flag)
                # msg = con.recv(buffer_size)
                if buffer_size>RECV_BUF_SIZE:
                    recv_size = 0
                    recv_msg = b''
                    while recv_size < buffer_size:
                        # recv_msg += conn.recv(RECV_BUF_SIZE)
                        recv_msg += conn.recv(min(RECV_BUF_SIZE, buffer_size-recv_size))
                        recv_size = len(recv_msg)
                else:
                    recv_msg = conn.recv(buffer_size)

                # print("msg3", len(recv_msg))
                buffer = struct.unpack(f"{buffer_size}s", recv_msg)[0]
                # print("flag",flag, len(buffer), buffer_size)
                # if flag>0:
                    # decode(buffer)
                filedir = os.path.dirname(savepath)
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                # print(path)
                with open(savepath, 'wb') as f:
                    f.write(buffer)  # 与readlines配对
                print(f"recved file: {savepath}")
                # print(f"recved file: {savepath:<50s} bufsize: {buffer_size:<10d}, recvsize: {len(recv_msg):<10d}")
            continue
        except Exception as e:
            print(e)
            exc = traceback.format_exc()
            print(exc)
            break
# 关闭服务器
# ser.close()

