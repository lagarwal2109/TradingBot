
import React from 'react';

interface ActionButtonProps {
  onClick?: () => void; // onClick is optional, especially if type="submit"
  children: React.ReactNode;
  className?: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'sidebar';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  isActive?: boolean; // For sidebar active state
  type?: 'button' | 'submit' | 'reset'; // Added type prop
  disabled?: boolean; // Added disabled prop
}

const ActionButton: React.FC<ActionButtonProps> = ({
  onClick,
  children,
  className = '',
  variant = 'primary',
  size = 'md',
  icon,
  isActive = false,
  type = 'button', // Default to 'button'
  disabled = false, // Default to false
}) => {
  const baseStyles = "font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background transition-all duration-150 ease-in-out inline-flex items-center";
  
  const sizeStyles = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg",
  };

  const getVariantStyles = () => {
    let styles = "";
    switch (variant) {
      case 'primary':
        styles = `bg-accent text-white ${disabled ? 'bg-accent/50 cursor-not-allowed' : 'hover:bg-accent-dark focus:ring-accent'} justify-center`;
        break;
      case 'secondary':
        styles = `bg-border-color text-text-primary ${disabled ? 'bg-border-color/50 text-text-secondary cursor-not-allowed' : 'hover:bg-card-header focus:ring-border-color'} justify-center`;
        break;
      case 'outline':
        styles = `bg-transparent border border-accent text-accent ${disabled ? 'border-accent/50 text-accent/50 cursor-not-allowed' : 'hover:bg-accent/10 focus:ring-accent'} justify-center`;
        break;
      case 'ghost':
        styles = `bg-transparent text-accent ${disabled ? 'text-accent/50 cursor-not-allowed' : 'hover:bg-accent/10 focus:ring-accent'} justify-center`;
        break;
      case 'sidebar':
        styles = `bg-transparent ${isActive ? 'text-accent font-semibold bg-accent/10' : 'text-text-secondary hover:text-text-primary hover:bg-border-color/50'} ${disabled ? 'text-text-secondary/50 cursor-not-allowed' : ''} focus:ring-accent w-full justify-start px-3 py-2.5 text-sm`;
        break;
      default:
        styles = `bg-accent text-white ${disabled ? 'bg-accent/50 cursor-not-allowed' : 'hover:bg-accent-dark focus:ring-accent'} justify-center`;
    }
    return styles;
  };
  
  const currentSizeStyle = variant === 'sidebar' ? '' : sizeStyles[size];

  return (
    <button
      type={type}
      onClick={onClick}
      className={`${baseStyles} ${currentSizeStyle} ${getVariantStyles()} ${className}`}
      disabled={disabled} // Pass disabled prop to button element
    >
      {icon && <span className={`flex-shrink-0 ${variant === 'sidebar' ? 'mr-3' : 'mr-2'}`}>{icon}</span>}
      <span className={`inline-flex items-center ${variant === 'sidebar' ? 'truncate' : ''}`}>{children}</span>
    </button>
  );
};

export default ActionButton;
