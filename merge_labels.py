import os
import yaml

def merge_labels():
    dataset_path = r'c:\projelerim\hut vol 2\dataset'
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    # 1. Update data.yaml
    if os.path.exists(data_yaml_path):
        print(f"Güncelleniyor: {data_yaml_path}...")
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        data['nc'] = 1
        data['names'] = ['uav']
        
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
        print("data.yaml başarıyla güncellendi.")
    else:
        print(f"Uyarı: data.yaml bulunamadı -> {data_yaml_path}")

    # 2. Update .txt files in train, valid, test folders
    # We'll check train, valid, and test directories
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        label_dir = os.path.join(dataset_path, subset, 'labels')
        
        if os.path.exists(label_dir):
            files = [f for f in os.listdir(label_dir) if f.endswith('.txt') and f != 'classes.txt']
            print(f"{label_dir} dizinindeki {len(files)} dosya işleniyor...")
            
            count = 0
            for filename in files:
                filepath = os.path.join(label_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    modified = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            if parts[0] != '0':
                                parts[0] = '0'
                                modified = True
                            new_lines.append(" ".join(parts) + "\n")
                    
                    if modified:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                        count += 1
                except Exception as e:
                    print(f"Hata: {filename} işlenemedi: {e}")
            
            print(f"{subset} tamamlandı: {count} dosya güncellendi.")
        else:
            print(f"Bilgi: {label_dir} bulunamadı, atlanıyor.")

if __name__ == "__main__":
    merge_labels()
