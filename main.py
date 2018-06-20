from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import  numpy as np
from keras.models import Model
import keras.callbacks as kcallbacks
from keras.layers import Dense,GlobalAveragePooling2D
import os
import keras
import skimage.io

model_path='./model/'
train_txt_path='./datasets/train.txt'
train_pic_path='./datasets/train/'
valid_txt_path='./datasets/test.txt'
valid_pic_path='./datasets/test/'
test_txt_path='./datesets/test_org.txt'
test_pic_path='./datasets/test/'
num_classes=100
finetune_epoch=100

#preload data
def get_image(txt_path,pic_path):
    f=open(file=txt_path,mode='r')
    lines=f.readlines()
    train_list=[]
    for line in lines:
        if pic_path is valid_pic_path:
            tmp = line.strip('\n').split(' ')
        else:
            tmp=line.strip('\n').split('\t')
        image_name=tmp[0]
        image_class=int(tmp[1])-1
        name_class=[image_name,image_class]
        train_list.append(name_class)
    random_order=np.random.permutation(train_list)
    train_data_list=[]
    train_label_list=[]
    for i in range(len(random_order)):
        image_name=random_order[i][0]
        image_label=random_order[i][1]
        image_data=image.load_img(pic_path+image_name,target_size=(224,224))
        #satisfy the parameter of preprocess_in_and_label
        image_data=image.img_to_array(image_data)
        #image_data=np.expand_dims(image_data,axis=0)
        train_data_list.append(image_data)
        train_label_list.append(image_label)
        print (i)
    f.close()
    return train_data_list,train_label_list

#preprocess input data and label output
def process_in_and_label(data_list,label_list):
    #satisfy the input parameter of image.ImageDataGenerator.flow
    data_list=np.array(data_list)
    label_list=np.array(label_list)
    label_list=keras.utils.to_categorical(label_list,num_classes)
    return data_list,label_list


if __name__ == '__main__':

    train_data_list, train_label_list=get_image(train_txt_path,train_pic_path)
    train_data_list,train_label_list=process_in_and_label(train_data_list,train_label_list)
    valid_data_list, valid_label_list=get_image(valid_txt_path,valid_pic_path)
    valid_data_list, valid_label_list=process_in_and_label(valid_data_list, valid_label_list)

    #total model
     #ResNet50(base_model)+changed fully connection layer（change units)=total model
    base_model=ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3))
    x=base_model.output
    #change top fully connection
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    predictions=Dense(100,activation='softmax')(x)
    #combine base_model and fully_connection layer
    model=Model(inputs=base_model.input,outputs=predictions)#model(total model）

    #define finetune model->setting training parameters->model configuration（optimization+loss+metric)->image.ImageDataGenerator（data augment way)->....flow（np,np)(data input for augment）
    #->begin trainning（model.fit_generator（))
    #train fully-connection layer
    for layer in base_model.layers:
        layer.trainable=False
    #configure the model
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    #train data augmentation
    train_gendata=image.ImageDataGenerator(
                                           featurewise_center=False,
                                           samplewise_center=False,
                                           featurewise_std_normalization=False,
                                           samplewise_std_normalization=False,
                                           zca_whitening=False,
                                           rotation_range=5,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.5,
                                           channel_shift_range=0.5,
                                           preprocessing_function=preprocess_input
                                          )
    #generate valid data(no augmentation)
    val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

    #augment train data in ImageDataGenerator way
    #flow（numpy,numpy,batchsize,seed,shuffle)
    train_generator =train_gendata.flow(train_data_list,train_label_list,batch_size=64,shuffle=True)
    val_generator=val_datagen.flow(valid_data_list,valid_label_list,batch_size=64,shuffle=False)

    #Save the model after every  five epoch.
    checkpoint=kcallbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5',period=5)
    #if the val_loss unchanged for 5 epochs the stop trainning process
    #earlystopping=kcallbacks.EarlyStopping(monitor='val_loss',patience=5)
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator)+1,
                        epochs=20,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers=2,
                        validation_steps=len(val_generator) + 1,
                        callbacks=[checkpoint])

    #decide appropriate conv layers to train
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    # train conv layer
    for layer in model.layers[:140]:
        layer.trainable=False
    for layer in model.layers[140:]:
        layer.trainable=True
    #configure the model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator)+1,
                        epochs=finetune_epoch,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers=2,
                        validation_steps=len(val_generator) + 1,
                        callbacks=[checkpoint])

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save(filepath=model_path+'model.h5')

    print('begin testing...')
    acc=model.evaluate(x=valid_data_list,y=valid_label_list,batch_size=100,verbose=0)
    print(acc)

    print('begin predict')
    f = open(test_txt_path)
    lines = f.readlines()
    flist = []
    for line in lines:
        temp = line.strip('\n')
        flist.append(temp)
    f.close()
    resultlist = []
    for filename in flist:
        img = skimage.io.imread(test_pic_path + filename)
        img = skimage.transform.resize(img, (224, 224), mode='edge', preserve_range=True)
        img = np.expand_dims(img, axis=0)
        # change
        img = preprocess_input(img)
        preds = model.predict(img)
        t_label_tmp = np.argmax(preds, 1)
        t_label = t_label_tmp[0]
        resultlist.append(filename + ' ' + str(int(t_label) + 1) + '\n')

    fresult = open('./datasets/submission.csv', 'w')
    for lin in resultlist:
        fresult.writelines(lin)
    fresult.close()















