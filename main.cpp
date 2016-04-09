#include "bpnn.hpp"

int main()
{
    const int       kInputNodeNum       = 2;
    const int       kOutputNodeNum      = 1;
    const char      *kBinaryNetFileName = "net.bin";
    const char      *kSampleFileName    = "sample.txt";
    BPNN<double>    bpnn;
    double          test_input[kInputNodeNum], test_output[kOutputNodeNum];

    if (bpnn.LoadNetFromBinaryFile(kBinaryNetFileName) == FALSE)
        bpnn.CreateNet(3, kInputNodeNum, 8, kOutputNodeNum);
    bpnn.LoadSampleFromFile(kSampleFileName);
    bpnn.Train(9999,0.0);
    bpnn.SaveNetToBinaryFile(kBinaryNetFileName);

    std::cout << "\n自定义测试模块，输入两个数字(0~1)进行测试，输入任意字母退出" << std::endl;
    std::cout << "输入：";
    while(std::cin >> test_input[0])
    {
        for (int i = 1; i < kInputNodeNum; i++)
            std::cin >> test_input[i];
        bpnn.TestSingleSample(test_input, test_output);
        std::cout << "输出：";
        for (int i = 0; i < kOutputNodeNum; i++)
            std::cout << test_output[0] << " ";
        std::cout << std::endl;
        std::cout << "输入：";
    }
    return 0;
}
