<?php 

class ThreadedCommand {
    
    public static function execute($directory, $chunk) {
        foreach ($chunk as $mesa) {
            $mesa = escapeshellarg($mesa);
            $file = $directory.$mesa.'.jpg';
            if(file_exists($file)) {
                printf("%s\n", $mesa. " Ya descargada");
                continue;
            }
            
            shell_exec("php command.php $directory $mesa > /dev/null 2> /dev/null &");
            print_r("Mesa $mesa Enviada a descargar\n");
        }
        
        return true;
    } 
}