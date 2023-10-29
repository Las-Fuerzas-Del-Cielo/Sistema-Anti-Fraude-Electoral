<?php
    
    ini_set('log_errors','On');
    ini_set('display_errors','Off');
    ini_set('error_reporting', E_ALL );
    define('WP_DEBUG', false);
    define('WP_DEBUG_LOG', true);
    define('WP_DEBUG_DISPLAY', false);

    require_once("./mesas.php");
    require_once("./threadedCommand.php");

    $directory = __DIR__.'/images/';
    @mkdir($directory, 0777, true);
    chmod($directory, 0777);
    $chunks = array_chunk($mesas,10, true);
    $dateSetDateStart = new DateTime("now");
    foreach($chunks as $chunk) {
        $dateStart = new DateTime("now");
        ThreadedCommand::execute($directory, $chunk);
        $dateEnd = new DateTime("now");
        $Partialdiff = (float) $dateEnd->diff($dateStart)->f;
        print_r("Tiempo en microsegundos: ".$Partialdiff."\n");
        
    }
    $endDataSetDate = new DateTime("now");
    echo 'fin, tiempo de ejecucion: ' . ($endDataSetDate->diff($dateSetDateStart)->format('H:i:s.u'));
    exit;
?>