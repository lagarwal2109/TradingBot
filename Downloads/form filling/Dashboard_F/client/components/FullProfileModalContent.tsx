import React from "react";
import { UserProfile } from "../types";

interface FullProfileModalContentProps {
  userProfile: UserProfile;
  T: any; // Translation object
}

function formatObjectForDisplay(obj: any): React.ReactNode {
  if (typeof obj !== "object" || obj === null) return String(obj);
  // Render as key-value pairs
  return (
    <ul className="list-disc list-inside text-sm text-text-secondary">
      {Object.entries(obj).map(([key, val]) => (
        <li key={key}>
          <span className="font-semibold">{key}:</span>{" "}
          {typeof val === "object" && val !== null
            ? JSON.stringify(val, null, 2)
            : String(val)}
        </li>
      ))}
    </ul>
  );
}

const ProfileDetailItem: React.FC<{
  label: string;
  value?: string | number | string[] | React.ReactNode | object | null;
  isList?: boolean;
}> = ({ label, value, isList }) => {
  // Always render the item, even if empty
  let displayValue: React.ReactNode;
  
  if (
    value === undefined ||
    value === null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0)
  ) {
    displayValue = <span className="text-gray-400 italic">Not provided</span>;
  }

  else if (isList && Array.isArray(value)) {
    displayValue = (
      <ul className="list-disc list-inside text-text-secondary">
        {value.map((item, index) => (
          <li key={index}>{String(item)}</li>
        ))}
      </ul>
    );
  } else if (Array.isArray(value)) {
    displayValue = value.join(", ");
  } else if (typeof value === "object" && !React.isValidElement(value)) {
    displayValue = formatObjectForDisplay(value);
  } else {
    displayValue = value;
  }

  return (
    <div className="bg-[#f8f5fc] p-3 sm:p-4 rounded-lg shadow-sm border border-border-color">
      <h4 className="text-sm font-semibold text-text-primary mb-0.5">
        {label}
      </h4>
      <div className="text-sm text-text-secondary">{displayValue}</div>
    </div>
  );
};

const FullProfileModalContent: React.FC<FullProfileModalContentProps> = ({
  userProfile,
  T,
}) => {
  const user = userProfile;
  const labels = T.fullProfileContent || {};

  // Fields to always show first, in order
  const primaryFields: (keyof typeof userProfile)[] = ["name", "age"];

  // Render primary fields (name, age) at the top
  const primaryItems = primaryFields.map((key) => {
    if (key === "name" && user.name) {
      return (
        <ProfileDetailItem
          key="name"
          label={labels.fullName || "Full Name"}
          value={user.name}
        />
      );
    }
    if (key === "age" && user.age !== undefined && user.age !== null) {
      return (
        <ProfileDetailItem
          key="age"
          label={labels.age || "Age"}
          value={user.age}
        />
      );
    }
    return null;
  });

  // Fields to skip from dynamic rendering - including the arrays that cause [object Object]
  const skipFields = new Set([
    "name", 
    "age", 
    "suggestedActions", // Skip this array to prevent [object Object]
    "documents"        // Skip this array to prevent [object Object]
  ]);

  // Dynamically render all other fields
  const dynamicItems = Object.entries(user)
    .filter(
      ([key]) =>
        !skipFields.has(key) // Only filter out skipFields, keep all others including empty/null
    )
    .map(([key, value]) => {
      // Try to use label translation if available, fallback to capitalized key
      const label =
        labels[key] ||
        key
          .replace(/([A-Z])/g, " $1")
          .replace(/^./, (str) => str.toUpperCase());

      // Special handling for certain fields
      if (key === "bio" || key === "onboardingSummary") {
        return (
          <ProfileDetailItem
            key={key}
            label={label}
            value={<span className="whitespace-pre-wrap">{String(value)}</span>}
          />
        );
      }
      if (Array.isArray(value)) {
        return (
          <ProfileDetailItem
            key={key}
            label={label}
            value={value}
            isList
          />
        );
      }
      return (
        <ProfileDetailItem
          key={key}
          label={label}
          value={value}
        />
      );
    });

  return (
    <div className="bg-[#f4f0ff] w-full h-full min-h-screen p-0 m-0">
      <div className="space-y-4">
        {primaryItems}
        {dynamicItems}
      </div>
    </div>
  );
};

export default FullProfileModalContent;
