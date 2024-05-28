#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>
#include <unistd.h>

#include "../io/io_manager.h" 
#include "../memory_pool/memory_pool.h"
#include "../uni_communication/uni_commu.h"
#include "../stream/pool.h"
#include "../common/common.h"

#define UNIX_PATH_MAX    108
#define SOCK_PATH        "/tmp/socket_path"

// Define the API request struct
typedef struct {
    int id; // Request ID
    int api_type; // API type: 0=I/O transfer, 1=memory allocation, 2=communication, 3=launching kernel
    char params[512]; // Parameters
} api_request;

// Depending on the API type, forward the request to different processes
void handle_api_request(api_request *req) {
    switch(req->api_type) {
        case 0:
            // Forward to IO Daemon
            break;
        case 1:
            // Forward to Fine-grain Memory Pool
            break;
        case 2:
            // Forward to Unified Communication Framework
            break;
        case 3:
            // Pass directly to the Stream Pool
            break;
        default:
            fprintf(stderr, "Unknown API type: %d\n", req->api_type);
    }
}

void *router(void *arg) {
    int *client_sock = (int *) arg;

    while(true) {
        api_request req = {0};
        int len = recv(*client_sock, &req, sizeof(req), 0);
        if (len <= 0) {
            break;
        }
        printf("recv: (id)%d (type)%s\n", req.id, req.api_type);
        handle_api_request(&req);
    }

    close(*client_sock);
    free(client_sock);
    return NULL;
}


int main(void) {
    struct sockaddr_un addr;
    int server_sock, client_sock, *new_sock;
    socklen_t client_addr_size;
    pthread_t thread_id;

    // Create socket
    if ((server_sock = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(-1);
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path)-1);
    unlink(SOCK_PATH);

    // Bind the socket to an address
    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("bind error");
        exit(-1);
    }

    // Start listening
    if (listen(server_sock, 5) == -1) {
        perror("listen error");
        exit(-1);
    }

    printf("Listening for connections...\n");
    client_addr_size = sizeof(struct sockaddr_un);

    while ((client_sock = accept(server_sock, (struct sockaddr*)&addr, &client_addr_size))) {
        puts("Connection accepted");

        new_sock = (int*)malloc(1);
        *new_sock = client_sock;

        if (pthread_create(&thread_id, NULL, router, (void*) new_sock) < 0) {
            perror("could not create thread");
            return 1;
        }

        pthread_detach(thread_id);
        printf("Handler assigned\n");
    }

    if (client_sock < 0) {
        perror("accept failed");
        return 1;
    }

    return 0;
}
