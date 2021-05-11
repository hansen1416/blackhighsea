/* The port number is passed as an argument */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <iostream>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

int main()
{
    // Create a socket
    int listening = socket(AF_INET, SOCK_STREAM, 0);
    if (listening == -1)
    {
        std::cerr << "Can't read a socket";
        return -1;
    }

    // Bind the socket to IP/port
    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(54000);
    // hint.sin_addr.s_addr = 
    inet_pton(AF_INET, "0.0.0.0", &hint.sin_addr); // 0.0.0.0 to docker ip

    if (bind(listening, (struct sockaddr*)&hint, sizeof(hint)) == -1 )
    {
        std::cerr << "Can't bind to IP / port";
        return -2;
    }

    // mark the socket for listening in
    if (listen(listening, SOMAXCONN) == -1)
    {
        std::cerr << "Can't listen";
        return -3;
    }

    // accept a call
    sockaddr_in client;
    socklen_t cleintsize = sizeof(client);
    char host[NI_MAXHOST];
    char svc[NI_MAXSERV];

    int clientsocket = accept(listening, (struct sockaddr*)&client, &cleintsize);

    if (clientsocket == -1) 
    {
        std::cerr << "Problem with client connecting";
        return -4;
    }

    // close the listening socket
    close(listening);

    memset(host, 0, NI_MAXHOST);
    memset(svc, 0, NI_MAXSERV);

    int result = getnameinfo((sockaddr*)&client, cleintsize, host, NI_MAXHOST, svc, NI_MAXSERV, 0);

    if (result) {
        std::cout << host << " connected on " << svc << std::endl;
    }
    else
    {
        inet_ntop(AF_INET, &client.sin_addr, host, NI_MAXHOST);
        std::cout << host << " connected on " << ntohs(client.sin_port) << std::endl;
    }

    // while receiving, display a message, echo message
    char buf[4096];
    while (true)
    {
        // clear the buffer
        memset(buf, 0, 4096); 
        // wait for the message
        int byterecv = recv(clientsocket, buf, 4096, 0);
        if (byterecv == -1) 
        {
            std::cerr << "There was a connection issue" << std::endl;
            break;
        }

        if (byterecv == 0) 
        {
            std::cerr << "client disconnected" << std::endl;
            break;
        }

        // display message
        std::cout << "Received " << std::string(buf, 0, byterecv) << std::endl;

        // resend a message
        send(clientsocket, buf, byterecv + 1, 0);
    }
    
    // close socket
    close(clientsocket);

    return 0;
}