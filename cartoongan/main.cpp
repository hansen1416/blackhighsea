#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// #include "cos_api.h"
// #include "cos_sys_config.h"
// #include "cos_defines.h"
#include <alibabacloud/oss/OssClient.h>

#include <iostream>
#include <memory>
#include <typeinfo>
#include <stdio.h>
#include <string.h> //strlen
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>    //close
#include <arpa/inet.h> //close
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros

int main(int argc, const char *argv[])
{
    /* 初始化OSS账号信息 */
    std::string AccessKeyId = "LTAI5tLwV38wLDsnsxKEdX3f";
    std::string AccessKeySecret = "vC8Uv3jophlVnRSkNBWkqTkp9fL9F7";
    std::string Endpoint = "oss-cn-hongkong.aliyuncs.com";
    /* 填写Bucket名称，例如examplebucket */
    std::string BucketName = "bhs-media";
    /* 填写文件完整路径，例如exampledir/exampleobject.txt。文件完整路径中不能包含Bucket名称 */
    std::string ObjectName = "imgs/exampleobject.txt";

    // Initialize the SDK
    AlibabaCloud::OSS::InitializeSdk();
    AlibabaCloud::OSS::ClientConfiguration conf;

    /* 设置连接池数，默认为16个。*/
    conf.maxConnections = 20;

    /* 设置请求超时时间，超时没有收到数据就关闭连接，默认为10000ms。*/
    conf.requestTimeoutMs = 8000;

    /* 设置建立连接的超时时间，默认为5000ms。*/
    conf.connectTimeoutMs = 8000;

    AlibabaCloud::OSS::OssClient client(Endpoint, AccessKeyId, AccessKeySecret, conf);

    std::shared_ptr<std::iostream> content = std::make_shared<std::stringstream>();
    *content << "Thank you for using Alibaba Cloud Object Storage Service!";
    AlibabaCloud::OSS::PutObjectRequest request(BucketName, ObjectName, content);
    
    /* 上传文件 */
    auto outcome = client.PutObject(request);
    
    if (!outcome.isSuccess()) {
        /* 异常处理 */
        std::cout << "PutObject fail" <<
        ",code:" << outcome.error().Code() <<
        ",message:" << outcome.error().Message() <<
        ",requestId:" << outcome.error().RequestId() << std::endl;
        /* 释放网络等资源。*/
        AlibabaCloud::OSS::ShutdownSdk();
        return -1;
    }

    std::string hostName = "https://bhs-media.oss-cn-hongkong.aliyuncs.com/";

    /* 释放网络等资源 */
    AlibabaCloud::OSS::ShutdownSdk();

    return 0;
}
