import React, { useState } from 'react';

const ToggleButton = () => {
  const [isSelected, setIsSelected] = useState(false);

  const toggleSelection = () => {
    setIsSelected(!isSelected);
  };

  const buttonStyle = {
    width: '24px',
    height: '24px',
    borderRadius: '6px',
    border: '2px solid #61439D', 
    background: isSelected ? '#61439D' : 'transparent',
    borderOpacity: isSelected ? 1 : 0,
  };

  return (
    <button onClick={toggleSelection} style={buttonStyle}>
      {isSelected ? '' : ''}
    </button>
  );
};

export default ToggleButton;
