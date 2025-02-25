import gc
import psutil
import yaml
from pathlib import Path

# Global variables for the ATLAS project
ATLAS_RUN_NUMBERS = ["00296939", "00296942", "00297447", "00297170", "00297041", "00297730", "00298591", "00298595", "00298609", "00298633", "00298687", "00298690", "00298771", "00298773", "00298862", "00298967", "00299055", "00299144", "00299147", "00299184", "00299241", "00299243", "00299278", "00299288", "00299315", "00299340", "00299343", "00299390", "00299584", "00300279", "00300287", "00300345", "00300415", "00300418", "00300487", "00300540", "00300571", "00300600", "00300655", "00300687", "00300784", "00300800", "00300863", "00300908", "00301912", "00301915", "00301918", "00301932", "00301973", "00302053", "00302137", "00302265", "00302269", "00302300", "00302347", "00302380", "00302391", "00302393", "00302737", "00302829", "00302831", "00302872", "00302919", "00302925", "00302956", "00303007", "00303059", "00303079", "00303201", "00303208", "00303264", "00303266", "00303291", "00303304", "00303338", "00303421", "00303499", "00303560", "00303638", "00303726", "00303811", "00303817", "00303819", "00303832", "00303846", "00303892", "00303943", "00304006", "00304008", "00304128", "00304178", "00304198", "00304211", "00304243", "00304308", "00304337", "00304409", "00304431", "00304494", "00305291", "00305293", "00305359", "00305380", "00305543", "00305571", "00305618", "00305671", "00305674", "00305723", "00305727", "00305735", "00305777", "00305811", "00305920", "00306247", "00306269", "00306278", "00306310", "00306384", "00306419", "00306442", "00306448", "00306451", "00306556", "00306655", "00306657", "00306714", "00307124", "00307126", "00307195", "00307259", "00307306", "00307354", "00307358", "00307394", "00307454", "00307514", "00307539", "00307569", "00307601", "00307619", "00307656", "00307710", "00307716", "00307732", "00307861", "00307935", "00308047", "00308084", "00309311", "00309314", "00309346", "00309375", "00309390", "00309440", "00309516", "00309640", "00309674", "00309759", "00310015", "00310210", "00310247", "00310249", "00310341", "00310370", "00310405", "00310468", "00310473", "00310574", "00310634", "00310691", "00310738", "00310781", "00310809", "00310863", "00310872", "00310969", "00311071", "00311170", "00311244", "00311287", "00311321", "00311365", "00311402", "00311473", "00311481"]

# Run numbers and their catalog counts
ATLAS_CATALOG_COUNTS = {                                                   
    '00296939': 21,
    '00296942': 45,
    '00297041': 36,
    '00297170': 90,
    '00297447': 52,
    '00297730': 109,
    '00298591': 21,
    '00298595': 105,
    '00298609': 38,
    '00298633': 69,
    '00298687': 100,
    '00298690': 16,
    '00298771': 37,
    '00298773': 33,
    '00298862': 135,
    '00298967': 585,
    '00299055': 112,
    '00299144': 83,
    '00299147': 74,
    '00299184': 266,
    '00299241': 21,
    '00299243': 26,
    '00299278': 36,
    '00299288': 89,
    '00299315': 19,
    '00299340': 21,
    '00299343': 37,
    '00299390': 8,
    '00299584': 975,
    '00300279': 33,
    '00300287': 8,
    '00300345': 33,
    '00300415': 93,
    '00300418': 14,
    '00300487': 202,
    '00300540': 271,
    '00300571': 142,
    '00300600': 86,
    '00300655': 304,
    '00300687': 299,
    '00300784': 17,
    '00300800': 416,
    '00300863': 274,
    '00300908': 229,
    '00301912': 18,
    '00301915': 14,
    '00301918': 72,
    '00301932': 420,
    '00301973': 794,
    '00302053': 1025,
    '00302137': 270,
    '00302265': 201,
    '00302269': 173,
    '00302300': 746,
    '00302347': 315,
    '00302380': 129,
    '00302391': 176,
    '00302393': 544,
    '00302737': 148,
    '00302829': 248,
    '00302831': 30,
    '00302872': 1025,
    '00302919': 71,
    '00302925': 172,
    '00302956': 288,
    '00303007': 17,
    '00303059': 81,
    '00303079': 145,
    '00303201': 164,
    '00303208': 766,
    '00303264': 59,
    '00303266': 390,
    '00303291': 187,
    '00303304': 535,
    '00303338': 1025,
    '00303421': 547,
    '00303499': 324,
    '00303560': 504,
    '00303638': 807,
    '00303726': 281,
    '00303811': 56,
    '00303817': 7,
    '00303819': 235,
    '00303832': 129,
    '00303846': 485,
    '00303892': 519,
    '00303943': 659,
    '00304006': 148,
    '00304008': 305,
    '00304128': 615,
    '00304178': 693,
    '00304198': 143,
    '00304211': 134,
    '00304243': 449,
    '00304308': 54,
    '00304337': 584,
    '00304409': 146,
    '00304431': 199,
    '00304494': 68,
    '00305291': 10,
    '00305293': 14,
    '00305359': 16,
    '00305380': 373,
    '00305543': 196,
    '00305571': 44,
    '00305618': 342,
    '00305671': 40,
    '00305674': 62,
    '00305723': 297,
    '00305727': 48,
    '00305735': 213,
    '00305777': 368,
    '00305811': 262,
    '00305920': 301,
    '00306247': 2,
    '00306269': 163,
    '00306278': 299,
    '00306310': 269,
    '00306384': 258,
    '00306419': 105,
    '00306442': 32,
    '00306448': 63,
    '00306451': 259,
    '00306556': 57,
    '00306655': 49,
    '00306657': 18,
    '00306714': 158,
    '00307124': 12,
    '00307126': 42,
    '00307195': 245,
    '00307259': 276,
    '00307306': 147,
    '00307354': 67,
    '00307358': 437,
    '00307394': 252,
    '00307454': 620,
    '00307514': 184,
    '00307539': 99,
    '00307569': 283,
    '00307601': 334,
    '00307619': 68,
    '00307656': 466,
    '00307710': 208,
    '00307716': 181,
    '00307732': 611,
    '00307861': 423,
    '00307935': 497,
    '00308047': 386,
    '00308084': 438,
    '00309311': 110,
    '00309314': 118,
    '00309346': 23,
    '00309375': 96,
    '00309390': 570,
    '00309440': 377,
    '00309516': 457,
    '00309640': 482,
    '00309674': 321,
    '00309759': 341,
    '00310015': 18,
    '00310210': 12,
    '00310247': 44,
    '00310249': 544,
    '00310341': 232,
    '00310370': 111,
    '00310405': 232,
    '00310468': 210,
    '00310473': 323,
    '00310574': 47,
    '00310634': 269,
    '00310691': 374,
    '00310738': 871,
    '00310781': 8,
    '00310809': 443,
    '00310863': 198,
    '00310872': 311,
    '00310969': 198,
    '00311071': 224,
    '00311170': 286,
    '00311244': 272,
    '00311287': 464,
    '00311321': 328,
    '00311365': 407,
    '00311402': 299,
    '00311473': 97,
    '00311481': 552,
}                                      

# Signal keys and their catalog file names
SIGNAL_CATALOGS = {
            "zprime": "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt",
            "wprime_qq": "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_Wprime_qq_3000",
            "zprime_bb": "mc20_13TeV_MC_Pythia8EvtGen_A14NNPDF23LO_Zprimebb3000"
            }

def get_catalog_count(run_number: str) -> int:
    """Get the number of catalogs for a run number from lookup table"""
    if run_number not in ATLAS_CATALOG_COUNTS:
        raise ValueError(f"No catalog count found for run number: {run_number}")
    return ATLAS_CATALOG_COUNTS[run_number]

# Memory usage
def get_system_usage():
        """Get current system memory and CPU usage with proper cleanup"""
        # Force garbage collection before checking memory
        gc.collect()
        
        # Get memory info
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval
        
        # Calculate memory in GB
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        
        return {
            'memory': {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'used_gb': used_gb,
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu_percent
            }
        }

def print_system_usage(prefix=""):
    """Print current system usage with optional prefix"""
    usage = get_system_usage()
    print(f"\n{prefix}System Usage:")
    print(f"Memory: {usage['memory']['used_gb']:.1f}GB / {usage['memory']['total_gb']:.1f}GB ({usage['memory']['percent']}%)")
    print(f"Available Memory: {usage['memory']['available_gb']:.1f}GB")
    print(f"CPU Usage: {usage['cpu']['percent']}%")


# Type conversion
from typing import Union, Dict, List, Any
import numpy as np
import tensorflow as tf

class TypeConverter:
    """Utility class for handling type conversions in the ML pipeline"""
    
    @staticmethod
    def to_numpy(data: Any, dtype=np.float32) -> np.ndarray:
        """Convert any numeric data to numpy array with specified dtype"""
        return np.array(data, dtype=dtype)
    
    @staticmethod
    def to_tensorflow(data: Any, dtype=tf.float32) -> tf.Tensor:
        """Convert any numeric data to tensorflow tensor with specified dtype"""
        return tf.convert_to_tensor(data, dtype=dtype)
    
    @staticmethod
    def to_python(data: Any) -> Union[int, float, List, Dict]:
        """Convert numpy/tensorflow types to Python native types"""
        if isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, (np.ndarray, tf.Tensor)):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: TypeConverter.to_python(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [TypeConverter.to_python(x) for x in data]
        return data 

class ConfigSerializer:
    """Handles serialization of configuration data"""
    
    @staticmethod
    def setup_yaml():
        """Configure YAML serializer for our needs"""
        def represent_none(dumper, _):
            return dumper.represent_scalar('tag:yaml.org,2002:null', '')
            
        # Add custom representers
        yaml.add_representer(type(None), represent_none)
        
        # Use safe dumper as base
        class SafeConfigDumper(yaml.SafeDumper):
            pass
            
        # Disable Python-specific tags
        SafeConfigDumper.ignore_aliases = lambda *args: True
        
        return SafeConfigDumper
    
    @staticmethod
    def to_yaml(data: Dict, file_path: Path):
        """Save configuration to YAML file"""
        # Convert Python types to simple types
        def simplify(obj):
            if isinstance(obj, (tuple, list)):
                return [simplify(x) for x in obj]
            elif isinstance(obj, dict):
                return {str(k): simplify(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # Prepare data
        simplified_data = simplify(data)
        
        # Save with custom dumper
        with open(file_path, 'w') as f:
            yaml.dump(
                simplified_data,
                f,
                Dumper=ConfigSerializer.setup_yaml(),
                default_flow_style=False,
                sort_keys=False
            )
    
    @staticmethod
    def from_yaml(file_path: Path) -> Dict:
        """Load configuration from YAML file"""
        with open(file_path) as f:
            return yaml.safe_load(f) 