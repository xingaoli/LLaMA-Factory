import json
import os
import re


def extract_data(root_path, save_path):

    with open(root_path, 'r') as f :#, \    
        train_file = json.load(f)

    test_data=dict()

    # TODO: convert the data into test data, containing the importance, multiple choice questions, graph questions
    for scene_id in train_file.keys():
        scene_data = train_file[scene_id]['key_frames']
        
        # for test file
        test_data[scene_id] = dict()
        test_data[scene_id]['key_frames'] = dict()

        for frame_id in scene_data.keys():
            frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            image_paths = scene_data[frame_id]['image_paths']

            # for test file
            test_data[scene_id]['key_frames'][frame_id] = dict()
            # test_data[scene_id]['key_frames'][frame_id]['key_object_infos'] = frame_data_infos
            test_data[scene_id]['key_frames'][frame_id]['QA'] = dict()
            test_data[scene_id]['key_frames'][frame_id]['image_paths'] = image_paths
            test_data[scene_id]['key_frames'][frame_id]['QA']['perception'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['planning'] = []
            test_data[scene_id]['key_frames'][frame_id]['QA']['behavior'] = []

            # get the classes of the important objects
            classes = []
            for obj_id in frame_data_infos.keys():
                obj_data = frame_data_infos[obj_id]
                classes.append(obj_data['Visual_description'].split('.')[0])
                print(classes)
            
            # get the location of the important objects
            locations = []
            for obj_id in frame_data_infos.keys():
                locations.append(obj_id)
                print(locations)
            
            # get the questions and answers of the perception
            perception = frame_data_qa["perception"]
            prediction = frame_data_qa["prediction"]
            planning = frame_data_qa["planning"]
            behavior = frame_data_qa["behavior"]
            # perception正常问题tag2,多选题tag0
            for qa in perception:
                question = qa['Q']
                answer = qa['A']

                # according to the classes to select the corresponding question
                flag = 0
                for cl in classes:
                    if cl.lower() in answer.lower():
                        flag = 1

                if 'describe' in question.lower():
                    flag = 1

                if 'identify' in question.lower():
                    flag = 1

                if 'Which lanes'.lower() in question.lower():
                    flag = 1

                if flag == 1:
                    qa['tag'] = [2]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                    # break
                
            # get the multiple choice questions and answers
            for qa in perception:
                question = qa['Q']
                answer = qa['A']

                flag = 0

                # if "What is the observed status of object".lower() in question.lower():
                #     flag = 1

                if "What is the moving status of object".lower() in question.lower():
                    flag = 1

                if flag == 1:
                    qa['tag'] = [0]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                    # break

            # prediction正常问答题 tag3; yes/no 问题 tag0
            # get the graph questions and answers
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']

                # according to the location to select the corresponding question
                flag = 0

                for loc in locations:
                    if loc.lower() in answer.lower():
                        flag = 1

                if 'future state' in question.lower():
                    flag = 1

                if flag == 1:
                    qa['tag'] = [3]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'].append(qa)
                    # break

            # get the yes or no questions and answers
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']
                if "yes" in answer.lower() or "no" in answer.lower():
                    qa['tag'] = [0]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['prediction'].append(qa)
                    # break

            # planning就只有三个问题，而且tag都是1
            # get the three questions from the planning "safe actions", "collision", ""
            # actions_question_added = False
            # collision_question_added = False
            # safe_actions_question_added = False


            for qa in planning:
                question = qa['Q']
                answer = qa['A']

                flag = 0

                if 'What actions could the ego vehicle take'.lower() in question.lower():
                    flag = 1

                if 'what is the most possible action' in question.lower():
                    flag = 1

                if 'What is the target action'.lower() in question.lower():
                    flag = 1

                if 'Based on the observation of'.lower() in question.lower():
                    flag = 1

                if "lead to a collision" in question.lower():
                    flag = 1

                if "safe actions" in question.lower():
                    flag = 1

                if flag == 1:
                    qa['tag'] = [1]
                    test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)

                # if "What actions could the ego vehicle take".lower() in question.lower() and not actions_question_added:
                #     qa['tag'] = [1]
                #     test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                #     actions_question_added = True
                # if "lead to a collision" in question.lower() and not collision_question_added:
                #     qa['tag'] = [1]
                #     test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                #     collision_question_added = True
                # if "safe actions" in question.lower() and not safe_actions_question_added:
                #     qa['tag'] = [1]
                #     test_data[scene_id]['key_frames'][frame_id]['QA']['planning'].append(qa)
                #     safe_actions_question_added = True
                # # Check if all question types have been added and exit the loop
                # if actions_question_added and collision_question_added and safe_actions_question_added:
                #     break
            
            for qa in behavior:
                question = qa['Q']
                answer = qa['A']
                qa['tag'] = [0]
                test_data[scene_id]['key_frames'][frame_id]['QA']['behavior'].append(qa)

    with open(save_path, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    # extract the data from the training json file
    root_path = "data/nuscenes_drivelm/v1_1_train_nus.json"
    save_path = "data/nuscenes_drivelm/trainval.json"
    extract_data(root_path, save_path)

    
