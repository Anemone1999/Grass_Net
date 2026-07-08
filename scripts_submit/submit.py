import subprocess
import tempfile
from pathlib import Path
from ruamel.yaml import YAML
import os

if __name__ == "__main__":
    template_path = "template.yaml"
    task_dict = {
        "ethanol_H_pred_qzvp_lm8": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_H_pred_qzvp_lm8.sh && sleep 600",
        "ethanol_H_pred_tzvp_lm6": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_H_pred_tzvp_lm6.sh && sleep 600",
        "ethanol_H_pred_svp_lm4": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_H_pred_svp_lm4.sh && sleep 600",

        "ethanol_X_pred_DM1_EA_1_svp_lm4": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_1_svp_lm4.sh && sleep 600",
        "ethanol_X_pred_DM1_EA_1_tzvp_lm6": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_1_tzvp_lm6.sh && sleep 600",

        "ethanol_X_pred_DM1_EA_1_qzvp_lm4": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_1_qzvp_lm4.sh && sleep 600",
        "ethanol_X_pred_DM1_EA_1_qzvp_lm6": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_1_qzvp_lm6.sh && sleep 600",
        "ethanol_X_pred_DM1_EA_1_qzvp_lm8": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_1_qzvp_lm8.sh && sleep 600",

        "ethanol_X_pred_DM1_EA_01": \
            "cd /root/code/SPHNet-main/scripts && bash ethanol_X_pred_DM1_EA_01.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_00": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_00.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_01": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_01.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_02": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_02.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_05": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_05.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_ODM_10": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_ODM_10.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_20": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_20.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_30": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_30.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_50": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_50.sh && sleep 600",
        "QH9id_X_pred_DM1_maemse_EA_00": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_maemse_EA_00.sh && sleep 600",
        "QH9id_X_pred_DM0_trHD_10_maemse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM0_trHD_10_maemse.sh && sleep 600",
        "QH9id_X_pred_DM1_trHD_01": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_trHD_01.sh && sleep 600",
        "QH9id_X_pred_DM1_trHD_01_maemse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_trHD_01_maemse.sh && sleep 600",
        "QH9id_X_pred_DM1_trHD_001_ft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_trHD_001_ft.sh && sleep 600",
        "QH9id_X_pred_DM1_trHD_005_ft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_trHD_005_ft.sh && sleep 600",
        "QH9id_X_pred_DM1_trHD_01_ft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_trHD_01_ft.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_001_ft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_001_ft.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_01_ft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_ft.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_001": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_001.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_01": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01.sh && sleep 600",

        "QH9id_X_pred_DM1_EA_10_trHD_01_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_001_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_001_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_005_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_005_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_05_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_05_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_10_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_10_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_20_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_20_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_50_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_50_double.sh && sleep 600",
        "QH9id_X_pred_DM0_EA_10_trHD_10_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM0_EA_10_trHD_10_double.sh && sleep 600",

        "QH9id_H_pred_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_H_pred_double.sh && sleep 600",
        "QH9id_H_pred_double_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_H_pred_double_nosparse.sh && sleep 600",
        "QH9id_H_pred_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_H_pred_nosparse.sh && sleep 600",
        "QH9id_H_pred_unidft_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_H_pred_unidft_nosparse.sh && sleep 600",
        "QH9id_H_pred_unidft_double_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_H_pred_unidft_double_nosparse.sh && sleep 600",
    
        "QH9id_X_pred_DM1_EA_10_trHD_01_rho_01_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_rho_01_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_01_rho_05_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_rho_05_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_01_rho_005_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_rho_005_double.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_01_rho_10_double": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_01_rho_10_double.sh && sleep 600",

        "QH9id_X_pred_DM1_double_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_double_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_double_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_double_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_trHD_10_double_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_trHD_10_double_nosparse.sh && sleep 600",

        "QH9id_X_pred_DM0_EA_10_ODM_10_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM0_EA_10_ODM_10_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_ODM_01_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_ODM_01_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_ODM_05_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_ODM_05_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_ODM_10_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_ODM_10_nosparse.sh && sleep 600",
        "QH9id_X_pred_DM1_EA_10_ODM_50_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_X_pred_DM1_EA_10_ODM_50_nosparse.sh && sleep 600",

        "NablaDFT_medium_H_pred_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_H_pred_nosparse.sh && sleep 600",
        "NablaDFT_medium_H_pred": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_H_pred.sh && sleep 600",
        "NablaDFT_medium_H_pred_unidft_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_H_pred_unidft_nosparse.sh && sleep 600",
        "NablaDFT_medium_H_pred_unidft": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_H_pred_unidft.sh && sleep 600",
            
        "NablaDFT_medium_X_pred_DM1_EA_10_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_X_pred_DM1_EA_10_nosparse.sh && sleep 600", 
        "NablaDFT_medium_X_pred_DM1_EA_10": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_X_pred_DM1_EA_10.sh && sleep 600",
        "NablaDFT_medium_X_pred_DM1_nosparse": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_X_pred_DM1_nosparse.sh && sleep 600",
        "NablaDFT_medium_X_pred_DM1": \
            "cd /root/code/SPHNet-main/scripts && bash NablaDFT_medium_X_pred_DM1.sh && sleep 600",

        "QH9id_QHNetX_pred_DM1_EA_10": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_QHNetX_pred_DM1_EA_10.sh && sleep 600", 
        "QH9id_QHNetH_pred_unidft": \
            "cd /root/code/SPHNet-main/scripts && bash QH9id_QHNetH_pred_unidft.sh && sleep 600", 
    }
    target_task_list = [
        # "QH9id_X_pred_DM1_trHD_01",
        # "NablaDFT_medium_X_pred_DM1_EA_10_trHD_10_double",
        # "NablaDFT_medium_X_pred_DM1_EA_10_double",
        # "NablaDFT_medium_X_pred_DM1_double",
        # "NablaDFT_medium_X_pred_DM1",
        # "NablaDFT_medium_H_pred_double",
        # "QH9id_X_pred_DM1_EA_10_ODM_10",
        # "QH9id_H_pred_nosparse",
        # "QH9id_QHNetX_pred_DM1_EA_10",
        # "NablaDFT_medium_X_pred_DM1_EA_10_nosparse",
        # "NablaDFT_medium_H_pred_unidft_nosparse",
        # "NablaDFT_medium_X_pred_DM1_nosparse",
        # "NablaDFT_medium_H_pred_nosparse",
        # "NablaDFT_medium_X_pred_DM1_EA_10",
        "NablaDFT_medium_H_pred_unidft",
        # "NablaDFT_medium_X_pred_DM1",
        # "NablaDFT_medium_H_pred",
        # "ethanol_X_pred_DM1_EA_1_qzvp_lm6",
        # "ethanol_X_pred_DM1_EA_1_qzvp_lm8",
        # "QH9id_X_pred_DM1_EA_10_ODM_05_nosparse",
        # "QH9id_X_pred_DM1_EA_10_ODM_10_nosparse",
        # "QH9id_X_pred_DM1_EA_10_ODM_50_nosparse",
        # "QH9id_X_pred_DM1_EA_10_double_nosparse",
        # "QH9id_X_pred_DM1_EA_10_trHD_10_double_nosparse",
        # "QH9id_X_pred_DM0_EA_10_trHD_10_double",
        # "QH9id_X_pred_DM1_EA_10_trHD_20_double",
        # "QH9id_X_pred_DM1_EA_10_trHD_50_double",
        # "QH9id_H_pred_double",
        # "QH9id_X_pred_DM1_EA_10_trHD_01_rho_005_double",
        # "QH9id_X_pred_DM1_EA_10_trHD_01_rho_10_double",
        # "QH9id_X_pred_DM1_EA_10_trHD_001",
        # "QH9id_X_pred_DM1_EA_10_trHD_01_ft",
        # "QH9id_X_pred_DM1_EA_10_trHD_01",
        # "QH9id_X_pred_DM1_EA_02",
    ]

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    for task in target_task_list:
        config['TaskName'] = task
        config['Entrypoint'] = task_dict[task]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmpf:
            yaml.dump(config, tmpf)
            temp_yaml_path = tmpf.name

        print(f'submitting task {task} under {temp_yaml_path}')
        task_cmd = f"volc ml_task submit -c {temp_yaml_path}"
        result = os.system(task_cmd)
        Path(temp_yaml_path).unlink(missing_ok=True)
