
CONNECTIVITY_CONFIG="connectivity.conf"

def get_ip_port():
    fd = open(CONNECTIVITY_CONFIG, "r")
    ip   = ''
    port = -1
    for line in fd:
        if (line[0] == '#') or (line == '') or ( not line ): continue
        else: 
            ip   = line.split(';')[0]
            port = int(line.split(';')[1])

    fd.close()

    if (ip == '') or (port == -1): 
        print("ERROR: The 'connectivity.conf' file has some error. Please, check it and try again")
        sys.exit(-1)

    return ip, port
