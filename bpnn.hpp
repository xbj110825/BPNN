#ifndef BPNET_HPP_INCLUDED
#define BPNET_HPP_INCLUDED

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <cstdarg>

#define BPNET_DEBUG
#define TRUE        1
#define FALSE       0
#define OK          1
#define ERROR       0
#define INFEASIBLE  -1
#define LOVERFLOW   -2
typedef int Status;

/*BPNN -- 反向传播神经网络 (Back Propagation Neural Network)*/
template <class ElemType>//ElemType一般设置为double或float两种浮点数之一
class BPNN
{
protected:
    /*可变参数*/
    ElemType    learning_rate;              //学习速率，取值范围为[0,1]，速率越大，拟合速度越快，但过大将导致无法收敛
    ElemType    learning_momentum;          //加速学习动量，取值范围为[0,1],以类似惯性的方式，加快学习速度

    /*网络结构*/
    int         layer_num;                  //神经网络层数
    int         *node_num_of_each_layer;    //各层节点数
    ElemType    ***weight;                  //节点间权重
    ElemType    ***delta_weight;            //节点间权重修正值
    ElemType    **node;                     //节点
    ElemType    **delta;                    //权值阈值修正相关中间变量

    /*训练样本*/
    int         sample_num;                 //样本数
    ElemType    **input;                    //训练样本输入值
    ElemType    **output;                   //训练样本输出值

    /*************************************************
    函数名称:FreeNet
    函数功能:释放delta,node,delta_weight,weight
            node_num_of_each_layer等指针所申请的动态空间
            并将各指针置为NULL
            layer_num归零
    被调清单:～Bpnet();
            Status CreateNetByArray(int layer_num, int *node_num_of_each_layer);
    *************************************************/
    void FreeNet();

    /*************************************************
    函数名称:FreeSample
    函数功能:释放input、output等指针所申请的动态空间
            并将各指针置为NULL
            sample_num不归零
    被调清单:～Bpnet();
            Status LoadSampleFromFile(const char *file_name);
    *************************************************/
    void FreeSample();

    /*************************************************
    函数名称:TrainSingleSample
    函数功能:使用单组训练样本训练进行单次训练
    调用清单:void *memcpy(void *dest, const void *src, size_t n);
    被调清单:ElemType TrainSingleTurn();
    输入参数:sample_index - 该训练样本在input与output中的索引
    函数返回:训练误差
    *************************************************/
    ElemType TrainSingleSample(int sample_index);

    /*************************************************
    函数名称:TrainSingleTurn
    函数功能:顺序使用所有训练样本进行一个轮次的训练
    调用清单:ElemType TrainSingleSample(int sample_index);
    被调清单:Status Train(int max_turn, ElemType desired_error);
    函数返回:本轮训练误差
    *************************************************/
    ElemType TrainSingleTurn();
public:
    /*************************************************
    函数名称:BPNN
    函数功能:将所有指针置为NULL，网络层数设置为0
            学习速率置为0.7，学习动量置为0.9
    *************************************************/
    BPNN();

    /*************************************************
    函数名称:～BPNN
    函数功能:释放动态申请的所有空间
    调用清单:void FreeNet();
            void FreeSample();
    *************************************************/
    ~BPNN();

    /*************************************************
    函数名称:get_learning_rate
    函数功能:获取当前学习速率
    函数返回:学习速率
    *************************************************/
    ElemType get_learning_rate() const;

    /*************************************************
    函数名称:get_learning_momentum
    函数功能:获取当前学习动量
    函数返回:学习动量
    *************************************************/
    ElemType get_learning_momentum() const;

    /*************************************************
    函数名称:set_learning_rate
    函数功能:设置学习速率
    输入参数:learning_rate - 学习速率
    函数返回:TRUE  - 新值合法，修改成功
            FALSE - 新值非法，修改失败
    *************************************************/
    Status set_learning_rate(ElemType learning_rate);

    /*************************************************
    函数名称:set_learning_momentum
    函数功能:设置学习动量
    输入参数:learning_momentum - 学习动量
    函数返回:TRUE  - 新值合法，修改成功
            FALSE - 新值非法，修改失败
    *************************************************/
    Status set_learning_momentum(ElemType learning_momentum);

    /*************************************************
    函数名称:CreateNet
    函数功能:创建有layer_num层的神经网络，每层网络的节点数值由后续函数参数决定
            例：CreateNet(3, 5, 8, 2);
            3层神经网络，第一层5节点，第二层8节点，第三层2节点
    调用清单:va_start();
            va_arg();
            va_end();
            Status CreateNetByArray(int layer_num, int *node_num_of_each_layer);
    输入参数:layer_num - 神经网络层数
            ...       - 每层节点数
    函数返回:TRUE  - 神经网络层数大于等于2，且各层节点数大于等于1
            FALSE - 参数非法
    其它说明:如果申请内存失败，将会在CreateNetByArray函数中调用函数exit(LOVERFLOW)
    *************************************************/
    Status CreateNet(int layer_num, ...);

    /*************************************************
    函数名称:CreateNetByArray
    函数功能:创建layer_num层的神经网络，第i层节点数为node_num_of_each_layer[i - 1]
    调用清单:void FreeNet();
            void srand(unsigned int seed);
            void exit(int status);
    被调清单:Status CreateNet(int layer_num, ...);
            Status LoadNetFromTextFile(const char *file_name);
            Status LoadNetFromBinaryFile(const char *file_name);
    输入参数:layer_num - 神经网络层数
            node_num_of_each_layer - 每层节点个数数组
    函数返回:TRUE  - 神经网络层数大于等于2，且各层节点数大于等于1
            FALSE - 参数非法
    其它说明:如果申请内存失败，将会调用函数exit(LOVERFLOW)
    *************************************************/
    Status CreateNetByArray(int layer_num, int *node_num_of_each_layer);

    /*************************************************
    函数名称:LoadSampleFromFile
    函数功能:从文件中加载训练样本
    调用清单:void open(const char* filename,int mode,int access);
            void close();
            void exit(int status);
    输入参数:file_name - 文件名
    函数返回:TRUE  - 文件加载成功
            FALSE - 文件加载失败
    其它说明:如果申请内存失败，将会调用函数exit(LOVERFLOW)
    *************************************************/
    Status LoadSampleFromFile(const char *file_name);

    /*************************************************
    函数名称:Train
    函数功能:使用所有训练样本，循环训练神经网络，直到达到（大于）最大训练轮次或者（小于）期望误差
    调用清单:ElemType TrainSingleTurn();
    输入参数:max_turn - 最大训练轮次
            desired_error - 期望误差
    函数返回:TRUE  - 训练轮次合法，即max_turn >= 1
            FALSE - 训练轮次非法
    其它说明:期望误差允许为负，当希望训练固定轮次，可将误差置为负数
            该函数至少会训练一次神经网络
    *************************************************/
    Status Train(int max_turn, ElemType desired_error);

    /*************************************************
    函数名称:TestSingleSample
    函数功能:将单组测试数据放入输入层，通过前向传递获取输出值
    调用清单:void *memcpy(void *dest, const void *src, size_t n);
    输入参数:test_input  - 测试样本输入值
    输出参数:test_output - 测试样本输出值
    *************************************************/
    void TestSingleSample(ElemType *test_input, ElemType *test_output);

    /*************************************************
    函数名称:SaveNetToTextFile
    函数功能:将神经网络的结构及权值以文本模式保存至指定文件中
    调用清单:void open(const char* filename,int mode,int access);
            void close();
    输入参数:file_name - 文件名
    函数返回:TRUE  - 写入成功
            FALSE - 写入失败
    *************************************************/
    Status SaveNetToTextFile(const char *file_name);

    /*************************************************
    函数名称:LoadNetFromTextFile
    函数功能:从指定文本文件中加载神经网络结构及权值
    调用清单:void open(const char* filename,int mode,int access);
            void close();
            void exit(int status);
            Status CreateNetByArray(int layer_num, int *node_num_of_each_layer);
    输入参数:file_name - 文件名
    函数返回:TRUE  - 加载成功
            FALSE - 加载失败
    其它说明:如果申请内存失败，将会调用函数exit(LOVERFLOW)
    *************************************************/
    Status LoadNetFromTextFile(const char *file_name);

    /*************************************************
    函数名称:SaveNetToBinaryFile
    函数功能:将神经网络的结构及权值以二进制模式保存至指定文件中
    调用清单:void open(const char* filename,int mode,int access);
            void close();
            write(char *buffer, streamsize size);
    输入参数:file_name - 文件名
    函数返回:TRUE  - 写入成功
            FALSE - 写入失败
    *************************************************/
    Status SaveNetToBinaryFile(const char *file_name);

    /*************************************************
    函数名称:LoadNetFromBinaryFile
    函数功能:从指二进制文件中加载神经网络结构及权值
    调用清单:void open(const char* filename,int mode,int access);
            void close();
            void exit(int status);
            read(char *buffer, streamsize size);
            Status CreateNetByArray(int layer_num, int *node_num_of_each_layer);
    输入参数:file_name - 文件名
    函数返回:TRUE  - 加载成功
            FALSE - 加载失败
    其它说明:如果申请内存失败，将会调用函数exit(LOVERFLOW)
    *************************************************/
    Status LoadNetFromBinaryFile(const char *file_name);
};

template <class ElemType>
void BPNN<ElemType>::FreeNet()
{
    //必须以申请顺序的逆序释放空间，因为释放后续申请的空间需要依赖layer_num以及node_num_of_each_layer的值
    if (delta != NULL)
    {
        for (int i = 1; i < layer_num; i++)
        {
            delete []delta[i];
            delta[i] = NULL;
        }
        delete []delta;
        delta = NULL;
    }
    if (node != NULL)
    {
        for (int i = 0; i < layer_num; i++)
        {
            delete []node[i];
            node[i] = NULL;
        }
        delete []node;
        node = NULL;
    }
    if (delta_weight != NULL)
    {
        for (int i = 0; i < layer_num - 1; i++)
        {
            for (int j = 0; j < node_num_of_each_layer[i + 1]; j++)
            {
                delete []delta_weight[i][j];
                delta_weight[i][j] = NULL;
            }
            delete []delta_weight[i];
            delta_weight[i] = NULL;
        }
        delete []delta_weight;
        delta_weight = NULL;
    }
    if (weight != NULL)
    {
        for (int i = 0; i < layer_num - 1; i++)
        {
            for (int j = 0; j < node_num_of_each_layer[i + 1]; j++)
            {
                delete []weight[i][j];
                weight[i][j] = NULL;
            }
            delete []weight[i];
            weight[i] = NULL;
        }
        delete []weight;
        weight = NULL;
    }
    if (node_num_of_each_layer != NULL)
    {
        delete []node_num_of_each_layer;
        node_num_of_each_layer = NULL;
    }
    layer_num = 0;
}

template <class ElemType>
void BPNN<ElemType>::FreeSample()
{
    //释放input,output的空间需要依赖sample_num的值
    if (input != NULL)
    {
        for (int i = 0; i < sample_num; i++)
        {
            delete []input[i];
            input[i] = NULL;
        }
        delete []input;
        input  = NULL;
    }
    if (output != NULL)
    {
        for (int i = 0; i < sample_num; i++)
        {
            delete []output[i];
            output[i] = NULL;
        }
        delete []output;
        output  = NULL;
    }
    sample_num = 0;
}

template <class ElemType>
ElemType BPNN<ElemType>::TrainSingleSample(int sample_index)
{
    int         i = 0, j = 0, k = 0;
    int         avoid_threshold = 1;//是否避开最后一个权值节点，1避开，0不避开
    ElemType    error = 0.0;

    /*正向传递-输入层*/
    memcpy(node[0], input[sample_index], (node_num_of_each_layer[0] - 1) * sizeof(ElemType));
    node[0][node_num_of_each_layer[0] - 1] = 1;//阈值

    /*正向传递-隐藏层&输出层*/
    for (i = 1; i < layer_num; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i] - 1; j++)
        {
            node[i][j] = 0;
            for (k = 0; k < node_num_of_each_layer[i - 1]; k++)
                node[i][j] += weight[i - 1][j][k] * node[i - 1][k];
            node[i][j] /= static_cast<ElemType>(node_num_of_each_layer[i - 1]);
            node[i][j] = 1.0/(1.0 + exp(-node[i][j]));
        }
        node[i][j] = 1; //阈值
    }

    /*反向传递-delta运算-输出层（最后一个节点为阈值，不用算，向前反馈的时候要注意避开）*/
    i = layer_num - 1;
    for (j = 0; j < node_num_of_each_layer[i] - 1; j++)
    {
        delta[i][j] = node[i][j] * (1 - node[i][j]) * (node[i][j] - output[sample_index][j]);
    }

    /*反向传递-delta运算-隐藏层&输入层*/
    avoid_threshold = 1;
    for (i = layer_num - 2; i > 0; i--)
    {
        for (k = 0; k < node_num_of_each_layer[i]; k++)
        {
            delta[i][k] = 0;
            for (j = 0; j < node_num_of_each_layer[i + 1] - avoid_threshold; j++)
            {
                delta[i][k] += delta[i + 1][j] * weight[i][j][k];
            }
            //delta[i][k] /= static_cast<ElemType>(node_num_of_each_layer[i + 1] - avoid_threshold);
            delta[i][k] *= node[i][j] * (1 - node[i][j]);
        }
        avoid_threshold = 0;
    }

    /*反向传递-修正权值和阈值*/
    for (i = 0; i < layer_num - 1; i++)
    {
        if (i == layer_num - 2)
            avoid_threshold = 1;
        for (j = 0; j < node_num_of_each_layer[i + 1] - avoid_threshold; j++)
        {
            for (k = 0; k < node_num_of_each_layer[i]; k++)
            {
                weight[i][j][k]         += learning_momentum * delta_weight[i][j][k];
                weight[i][j][k]         -= learning_rate * delta[i + 1][j] * node[i][k];
                delta_weight[i][j][k]   =  learning_momentum * delta_weight[i][j][k] -
                                           learning_rate * delta[i + 1][j] * node[i][k];
            }
        }
    }

    /*误差计算*/
    for (i = 0; i < node_num_of_each_layer[layer_num - 1] - 1; i++)
        error += (node[layer_num - 1][i] - output[sample_index][i]) *
                 (node[layer_num - 1][i] - output[sample_index][i]);
    error /= 2.0;
    return error;
}

template <class ElemType>
ElemType BPNN<ElemType>::TrainSingleTurn()
{
    int     i = 0;
    ElemType   error = 0.0;
    for (i = 0; i < sample_num; i++)
        error += TrainSingleSample(i);
    error /= sample_num;
    return error;
}

template <class ElemType>
BPNN<ElemType>::BPNN()
{
    layer_num               = 0;
    node_num_of_each_layer  = NULL;
    weight                  = NULL;
    delta_weight            = NULL;
    node                    = NULL;
    delta                   = NULL;
    learning_rate           = 0.7;
    learning_momentum       = 0.9;
    input                   = NULL;
    output                  = NULL;
}

template <class ElemType>
BPNN<ElemType>::~BPNN()
{
    FreeNet();
    FreeSample();
}

template <class ElemType>
ElemType BPNN<ElemType>::get_learning_rate() const
{
    return learning_rate;
}

template <class ElemType>
ElemType BPNN<ElemType>::get_learning_momentum() const
{
    return learning_momentum;
}

template <class ElemType>
Status BPNN<ElemType>::set_learning_rate(ElemType learning_rate)
{
    if (learning_rate > 1.0 || learning_rate < 0.0)
    {
        std::cout << "ERROR: learning_rate > 1.0 || learning_rate < 0.0" << std::endl;
        return FALSE;
    }
    else
    {
        this->learning_rate = learning_rate;
        return TRUE;
    }
}

template <class ElemType>
Status BPNN<ElemType>::set_learning_momentum(ElemType learning_momentum)
{
    if (learning_momentum > 1.0 || learning_momentum < 0.0)
    {
        std::cout << "ERROR: learning_momentum > 1.0 || learning_momentum < 0.0" << std::endl;
        return FALSE;
    }
    else
    {
        this->learning_momentum = learning_momentum;
        return TRUE;
    }
}

template <class ElemType>
Status BPNN<ElemType>::CreateNet(int layer_num, ...)
{
    if (layer_num < 2)
    {
        std::cout << "ERROR: layer_num < 2" << std::endl;
        return FALSE;
    }

    Status return_value;
    int *tmp = new int[layer_num];
    va_list arg_ptr;
    va_start(arg_ptr, layer_num);
    for (int i = 0; i < layer_num; i++)
        tmp[i] = va_arg(arg_ptr, int);
    va_end(arg_ptr);
    return_value = CreateNetByArray(layer_num, tmp);
    delete []tmp;

    return return_value;
}

template <class ElemType>
Status BPNN<ElemType>::CreateNetByArray(int layer_num, int *node_num_of_each_layer)
{
    int i = 0, j = 0, k = 0;

    /*如果曾经申请过空间，则释放空间*/
    FreeNet();

    /*设置随机种子，在初始化权重时，随机生成*/
    srand(static_cast<unsigned int>(time(0)));

    /*layer_num赋值*/
    if (layer_num < 2)
    {
        std::cout << "ERROR: layer_num < 2" << std::endl;
        return FALSE;
    }
    this->layer_num = layer_num;

    /*node_num_of_each_layer赋值*/
    this->node_num_of_each_layer = new int[this->layer_num];
    if (this->node_num_of_each_layer == NULL)
        exit(LOVERFLOW);
    for (i = 0; i < this->layer_num; i++)
    {
        if (node_num_of_each_layer[i] < 1)
        {
            std::cout << "ERROR: node_num_of_each_layer[" << i << "] < 1" << std::endl;
            return FALSE;
        }
        this->node_num_of_each_layer[i] = node_num_of_each_layer[i] + 1;  ///每层节点数增加一，作为阈值，该点输入始终为1
    }

    /*weight[i][j][k]代表第i层的第k个元素和第i+1层的第j个元素之间的权重*/
    weight          = new ElemType**[this->layer_num - 1];
    delta_weight    = new ElemType**[this->layer_num - 1];
    if (weight == NULL || delta_weight == NULL)
        exit(LOVERFLOW);
    for (i = 0; i < this->layer_num - 1; i++)
    {
        weight[i]       = new ElemType*[this->node_num_of_each_layer[i + 1]];
        delta_weight[i] = new ElemType*[this->node_num_of_each_layer[i + 1]];
        if (weight[i] == NULL || delta_weight[i] == NULL)
            exit(LOVERFLOW);

        for (j = 0; j < this->node_num_of_each_layer[i + 1]; j++)
        {
            weight[i][j]        = new ElemType[this->node_num_of_each_layer[i]];
            delta_weight[i][j]  = new ElemType[this->node_num_of_each_layer[i]];
            if (weight[i][j] == NULL || delta_weight[i][j] == NULL)
                exit(LOVERFLOW);

            /*初始权值 [-1, 1]*/
            for (k = 0; k < this->node_num_of_each_layer[i]; k++)
            {
                do
                {
                    weight[i][j][k] = (rand()/(RAND_MAX + 0.0)) * 2 - 1;
                } while(fabs(weight[i][j][k]) < 1e-6);
                delta_weight[i][j][k] = 0.0;
            }
        }
    }

    /*node[i][j]代表第i层第j个元素*/
    node = new ElemType*[this->layer_num];
    if (node == NULL)
        exit(LOVERFLOW);
    for (i = 0; i < this->layer_num; i++)
    {
        node[i] = new ElemType[this->node_num_of_each_layer[i]];
        if (node[i] == NULL)
            exit(LOVERFLOW);
    }

    /*delta*/
    delta = new ElemType*[this->layer_num];
    if (delta == NULL)
        exit(LOVERFLOW);
    delta[0] = NULL;
    for (i = 1; i < this->layer_num; i++)
    {
        delta[i] = new ElemType[this->node_num_of_each_layer[i]];
        if (delta[i] == NULL)
            exit(LOVERFLOW);
    }
    return TRUE;
}

template <class ElemType>
Status BPNN<ElemType>::LoadSampleFromFile(const char *file_name)
{
    int input_node_num  = 0;
    int output_node_num = 0;
    int i = 0, j = 0;

    std::ifstream infile;
    infile.open(file_name, std::ios::in);
    if (!infile)
    {
        std::cout << "打开文件" << file_name << "失败" << std::endl;
        return FALSE;
    }

    FreeSample();
    infile >> sample_num;
    if (sample_num < 1)
    {
        std::cout << "ERROR: sample_num < 1" << std::endl;
        sample_num = 0;
        return FALSE;
    }

    infile >> input_node_num >> output_node_num;
    if (input_node_num < 1 || output_node_num < 1)
    {
        std::cout << "ERROR: input_node_num < 1 || output_node_num < 1" << std::endl;
        return FALSE;
    }


    input   = new ElemType*[sample_num];
    output  = new ElemType*[sample_num];
    if (input == NULL || output == NULL)
        exit(LOVERFLOW);

    std::cout << "正在从" << file_name << "文件加载训练样本，请稍后..." << std::endl;
    for (i = 0; i < sample_num; i++)
    {
        input[i]    = new ElemType[input_node_num];
        output[i]   = new ElemType[output_node_num];
        if (input[i] == NULL || output[i] == NULL)
            exit(LOVERFLOW);
        for (j = 0; j < input_node_num; j++)
            infile >> input[i][j];
        for (j = 0; j < output_node_num; j++)
            infile >> output[i][j];
    }
    infile.close();
    std::cout << "训练样本加载完成" << std::endl;
    return TRUE;
}

template <class ElemType>
Status BPNN<ElemType>::Train(int max_turn, ElemType desired_error)
{
    int         i = 0;
    ElemType    error = 0.0;

    if (max_turn < 1)
    {
        std::cout << "ERROR: max_turn < 1" << std::endl;
        return FALSE;
    }

    std::cout << "正在训练，请稍后..." << std::endl;
    for (i = 0; i < max_turn; i++)
    {
        error = TrainSingleTurn();
        std::cout << "第" << std::setw(8) << i + 1 << "轮训练 - 本轮误差：" << std::setw(16) << error << std::endl;
        if (error < desired_error)
        {
            std::cout << "已达期望误差" << std::endl;
            return TRUE;
        }
    }

    std::cout << "已达最大训练轮次" << std::endl;
    return TRUE;
}

template <class ElemType>
void BPNN<ElemType>::TestSingleSample(ElemType *test_input, ElemType *test_output)
{
    int i, j, k;

    /*正向传递-输入层*/
    memcpy(node[0], test_input, (node_num_of_each_layer[0] - 1) * sizeof(ElemType));
    node[0][node_num_of_each_layer[0] - 1] = 1;//阈值

    /*正向传递-隐藏层&输出层*/
    for (i = 1; i < layer_num; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i] - 1; j++)
        {
            node[i][j] = 0;
            for (k = 0; k < node_num_of_each_layer[i - 1]; k++)
                node[i][j] += weight[i - 1][j][k] * node[i - 1][k];
            node[i][j] /= static_cast<ElemType>(node_num_of_each_layer[i - 1]);
            node[i][j] = 1.0/(1.0 + exp(-node[i][j]));
        }
        node[i][j] = 1; //阈值
    }

    memcpy(test_output, node[layer_num - 1], (node_num_of_each_layer[layer_num - 1] - 1) * sizeof(ElemType));
}

template <class ElemType>
Status BPNN<ElemType>::SaveNetToTextFile(const char *file_name)
{
    int i = 0, j = 0, k = 0;

    std::ofstream outfile;
    outfile.open(file_name, std::ios::out);
    if (!outfile)
    {
        std::cout << "写入文件" << file_name << "失败" << std::endl;
        return FALSE;
    }

    outfile << learning_rate << std::endl;
    outfile << learning_momentum << std::endl;
    outfile << layer_num << std::endl;

    for (i = 0; i < layer_num; i++)
        outfile << node_num_of_each_layer[i] << " ";
    outfile << std::endl;

    std::cout << "正在将神经网络结构及参数保存至" << file_name << "文件，请稍后..." << std::endl;
    for (i = 0; i < layer_num - 1; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i + 1]; j++)
        {
            for (k = 0; k < node_num_of_each_layer[i]; k++)
            {
                outfile << weight[i][j][k] << " " << delta_weight[i][j][k] << " ";
            }
        }
    }
    outfile << std::endl;
    outfile.close();
    std::cout << "保存成功" << std::endl;
    return TRUE;
}

template <class ElemType>
Status BPNN<ElemType>::LoadNetFromTextFile(const char *file_name)
{
    int i, j, k;
    int *new_node_num_of_each_layer;

    std::ifstream infile;
    infile.open(file_name, std::ios::in);
    if (!infile)
    {
        std::cout << "读取文件" << file_name << "失败" << std::endl;
        return FALSE;
    }

    infile >> learning_rate;
    infile >> learning_momentum;
    infile >> layer_num;

    new_node_num_of_each_layer = new int[layer_num];
    if (new_node_num_of_each_layer == NULL)
        exit(LOVERFLOW);

    for (i = 0; i < layer_num; i++)
    {
        infile >> new_node_num_of_each_layer[i];
        new_node_num_of_each_layer[i]--; //去掉阈值多带的节点
    }
    CreateNetByArray(layer_num, new_node_num_of_each_layer);

    std::cout << "正在从" << file_name << "文件加载神经网络结构及参数，请稍后..." << std::endl;
    for (i = 0; i < layer_num - 1; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i + 1]; j++)
        {
            for (k = 0; k < node_num_of_each_layer[i]; k++)
            {
                infile >> weight[i][j][k] >> delta_weight[i][j][k];
            }
        }
    }
    infile.close();
    std::cout << "加载成功" << std::endl;
    return TRUE;
}

template <class ElemType>
Status BPNN<ElemType>::SaveNetToBinaryFile(const char *file_name)
{
    int i = 0, j = 0, k = 0;

    std::ofstream outfile;
    outfile.open(file_name, std::ios::binary);
    if (!outfile)
    {
        std::cout << "写入文件" << file_name << "失败" << std::endl;
        return FALSE;
    }

    outfile.write((char*)(&learning_rate), sizeof(ElemType));
    outfile.write((char*)(&learning_momentum), sizeof(ElemType));
    outfile.write((char*)(&layer_num), sizeof(int));

    for (i = 0; i < layer_num; i++)
        outfile.write((char*)(&(node_num_of_each_layer[i])), sizeof(int));

    std::cout << "正在将神经网络结构及参数保存至" << file_name << "文件，请稍后..." << std::endl;
    for (i = 0; i < layer_num - 1; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i + 1]; j++)
        {
            for (k = 0; k < node_num_of_each_layer[i]; k++)
            {
                outfile.write((char*)(&(weight[i][j][k])), sizeof(ElemType));
                outfile.write((char*)(&(delta_weight[i][j][k])), sizeof(ElemType));
            }
        }
    }
    outfile.close();
    std::cout << "保存成功" << std::endl;
    return TRUE;
}

template <class ElemType>
Status BPNN<ElemType>::LoadNetFromBinaryFile(const char *file_name)
{
    int i, j, k;
    int *new_node_num_of_each_layer;

    std::ifstream infile;
    infile.open(file_name, std::ios::binary);
    if (!infile)
    {
        std::cout << "读取文件" << file_name << "失败" << std::endl;
        return FALSE;
    }

    infile.read((char*)(&learning_rate), sizeof(ElemType));
    infile.read((char*)(&learning_momentum), sizeof(ElemType));
    infile.read((char*)(&layer_num), sizeof(int));

    new_node_num_of_each_layer = new int[layer_num];
    if (new_node_num_of_each_layer == NULL)
        exit(LOVERFLOW);

    for (i = 0; i < layer_num; i++)
    {
        infile.read((char*)(&(new_node_num_of_each_layer[i])), sizeof(int));
        new_node_num_of_each_layer[i]--; //去掉阈值多带的节点
    }
    CreateNetByArray(layer_num, new_node_num_of_each_layer);

    std::cout << "正在从" << file_name << "文件加载神经网络结构及参数，请稍后..." << std::endl;
    for (i = 0; i < layer_num - 1; i++)
    {
        for (j = 0; j < node_num_of_each_layer[i + 1]; j++)
        {
            for (k = 0; k < node_num_of_each_layer[i]; k++)
            {
                infile.read((char*)(&(weight[i][j][k])), sizeof(ElemType));
                infile.read((char*)(&(delta_weight[i][j][k])), sizeof(ElemType));
            }
        }
    }
    infile.close();
    std::cout << "加载成功" << std::endl;
    return TRUE;
}

#endif
